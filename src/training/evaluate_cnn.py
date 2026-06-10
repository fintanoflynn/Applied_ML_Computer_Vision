import time
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.color_dataset import PlantVillageDataset, image_paths_and_labels, classes
from src.training.train_cnn import (
    BATCH_SIZE,
    MODEL_SAVE_PATH,
    compute_mean_std,
    evaluate,
    make_class_weights,
    make_transforms,
    split_data,
)
from src.models.cnn_scratch import CNN


def measure_inference_time(model, n_runs=100):
    cpu_model = model.cpu().eval()
    dummy = torch.zeros(1, 3, 224, 224)
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            cpu_model(dummy)
            times.append(time.perf_counter() - start)
    return sum(times) / len(times) * 1000


def per_crop_macro_f1(per_class_f1):
    crop_f1s = defaultdict(list)
    for cls, f1 in zip(classes, per_class_f1):
        crop = cls.split("___")[0]
        crop_f1s[crop].append(f1)
    return {crop: sum(f1s) / len(f1s) for crop, f1s in sorted(crop_f1s.items())}


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    train_samples, _, test_samples = split_data(image_paths_and_labels)

    mean, std = compute_mean_std(train_samples)
    _, eval_transform = make_transforms(mean, std)

    test_loader = DataLoader(
        PlantVillageDataset(test_samples, transform=eval_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = CNN().to(device)
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    class_weights = make_class_weights(train_samples, len(classes), device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    test_loss, test_acc, macro_f1, per_class_f1, per_class_precision, per_class_recall, conf_matrix = evaluate(model, test_loader, loss_fn, device)

    inference_ms = measure_inference_time(model)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test acc: {test_acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Inference time: {inference_ms:.2f} ms/image (CPU)")
    print()
    print(f"{'Class':<45} {'F1':>6} {'Precision':>10} {'Recall':>8}")
    print("-" * 72)
    for i, cls in enumerate(classes):
        print(f"{cls:<45} {per_class_f1[i]:>6.4f} {per_class_precision[i]:>10.4f} {per_class_recall[i]:>8.4f}")

    print()
    print(f"{'Crop':<30} {'Macro F1':>8}")
    print("-" * 40)
    for crop, f1 in per_crop_macro_f1(per_class_f1).items():
        print(f"{crop:<30} {f1:>8.4f}")

    print()
    print("Confusion matrix:")
    print(conf_matrix)


if __name__ == "__main__":
    main()
