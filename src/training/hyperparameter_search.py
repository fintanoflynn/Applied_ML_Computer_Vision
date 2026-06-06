import random
import math
import json
import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from src.data.dataset import stratified_splits
from src.data.transforms import build_train_transform, build_eval_transform
from src.models.resnet_transfer import build_resnet18
from src.utils.seed import set_seed


def sample_hyperparameters(rng):
    
    log_lr = rng.uniform(math.log(1e-5), math.log(1e-2))
    lr = math.exp(log_lr)

    weight_decay = rng.choice([0, 1e-4, 1e-3])
    dropout = rng.choice([0.1, 0.3, 0.5])
    augmentation = rng.choice(["light", "heavy"])

    return {
        "lr": lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "augmentation": augmentation
    }


def run_trial(trial_num, params, data_root, num_epochs, device):

    # Recreate train dataset with correct augmentation for this trial
    train_ds, val_ds, _ = stratified_splits(
        data_root,
        train_transform=build_train_transform(augmentation=params["augmentation"]),
        eval_transform=build_eval_transform()
    )
    
    print(f"\n--- Trial {trial_num} ---")
    print(f"  lr={params['lr']:.2e}  weight_decay={params['weight_decay']}  "
          f"dropout={params['dropout']}  augmentation={params['augmentation']}")

    # Rebuild dataloaders with correct augmentation strength
    train_loader = DataLoader(
        train_ds, batch_size=64, shuffle=True,
        num_workers=2, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=64, shuffle=False, num_workers=2
    )

    # Build model
    model = build_resnet18(
        num_classes=len(train_ds.classes),
        finetune_mode="head_only",
        dropout=params["dropout"]
    )
    model = model.to(device)

    # Class weighted loss
    counts = train_ds.class_counts().astype(np.float64)
    weights = counts.sum() / (len(counts) * counts)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimiser
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(
        trainable,
        lr=params["lr"],
        weight_decay=params["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=num_epochs
    )

    best_val_f1 = 0.0
    patience_counter = 0
    patience = 5

    for epoch in range(1, num_epochs + 1):

        # Training phase
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimiser.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

        # Validation phase
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_true.extend(labels.numpy())

        val_macro_f1 = f1_score(val_true, val_preds, average='macro')
        scheduler.step()

        print(f"  Epoch {epoch}/{num_epochs} | val_macro_f1={val_macro_f1:.4f}")

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"  Trial {trial_num} best val_macro_f1={best_val_f1:.4f}")
    return best_val_f1


def hyperparameter_search(
    data_root="data/raw",
    n_trials=10,
    epochs_per_trial=10,
    seed=42
):
    set_seed(seed)
    rng = random.Random(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = []

    for trial in range(1, n_trials + 1):
        params = sample_hyperparameters(rng)
        best_f1 = run_trial(
            trial, params, data_root, epochs_per_trial,
            device
        )
        results.append({
            "trial": trial,
            "params": params,
            "best_val_macro_f1": best_f1
        })

    # Sort by best f1
    results.sort(key=lambda x: x["best_val_macro_f1"], reverse=True)

    print("\n=== Hyperparameter Search Results ===")
    for r in results:
        print(f"Trial {r['trial']:2d} | "
              f"f1={r['best_val_macro_f1']:.4f} | "
              f"lr={r['params']['lr']:.2e} | "
              f"wd={r['params']['weight_decay']} | "
              f"dropout={r['params']['dropout']} | "
              f"aug={r['params']['augmentation']}")

    best = results[0]
    print(f"\nBest trial: {best['trial']} with val_macro_f1={best['best_val_macro_f1']:.4f}")
    print(f"Best params: {best['params']}")

    # Save results to JSON
    Path("runs").mkdir(exist_ok=True)
    with open("runs/hp_search_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to runs/hp_search_results.json")

    if os.path.exists("/content/drive/MyDrive"):
        shutil.copy("runs/hp_search_results.json",
                    "/content/drive/MyDrive/hp_search_results.json")
        print("Results backed up to Google Drive")

    return best["params"]


if __name__ == "__main__":
    best_params = hyperparameter_search()