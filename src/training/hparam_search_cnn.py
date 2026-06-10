import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.color_dataset import PlantVillageDataset, image_paths_and_labels, classes
from src.models.cnn_scratch import CNN
from src.training.train_cnn import (
    BATCH_SIZE,
    RANDOM_SEED,
    compute_mean_std,
    evaluate,
    make_class_weights,
    make_transforms,
    split_data,
    train_one_epoch,
)

N_TRIALS = 20
EPOCHS = 10 
PATIENCE = 3 
RESULTS_PATH = Path("models/hparam_search_cnn.json")


def sample_config(rng: random.Random) -> dict:
    log_lr = rng.uniform(math.log(1e-5), math.log(1e-2))
    return {
        "learning_rate": math.exp(log_lr),
        "weight_decay": rng.choice([0, 1e-4, 1e-3]),
        "dropout": rng.choice([0.1, 0.3, 0.5]),
        "aug_strength": rng.choice(["light", "heavy"]),
    }


def run_trial(config: dict, train_samples, val_samples, mean, std, device) -> float:
    train_transform, eval_transform = make_transforms(mean, std, config["aug_strength"])

    train_loader = DataLoader(
        PlantVillageDataset(train_samples, transform=train_transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        PlantVillageDataset(val_samples, transform=eval_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = CNN(dropout=config["dropout"]).to(device)
    class_weights = make_class_weights(train_samples, len(classes), device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    def lr_lambda(epoch):
        if epoch < 1:
            return 0.1
        progress = (epoch - 1) / max(1, EPOCHS - 2)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_f1 = 0.0
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        _, _, val_f1, _, _, _, _ = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            break

    return best_val_f1


def main():
    torch.manual_seed(RANDOM_SEED)
    rng = random.Random(RANDOM_SEED)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    train_samples, val_samples, _ = split_data(image_paths_and_labels)
    mean, std = compute_mean_std(train_samples)
    results = []

    for trial in range(N_TRIALS):
        config = sample_config(rng)
        print(f"\nTrial {trial + 1}/{N_TRIALS} | config: {config}")

        val_f1 = run_trial(config, train_samples, val_samples, mean, std, device)
        results.append({"trial": trial + 1, "config": config, "val_macro_f1": val_f1})
        print(f"Trial {trial + 1} best val macro-F1: {val_f1:.4f}")

    results.sort(key=lambda r: r["val_macro_f1"], reverse=True)
    best = results[0]

    print(f"\nBest config: {best['config']}")
    print(f"Best val macro-F1: {best['val_macro_f1']:.4f}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"All results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
