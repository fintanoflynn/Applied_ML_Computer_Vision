from pathlib import Path

import math

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from src.data.color_dataset import PlantVillageDataset, image_paths_and_labels, classes
from src.models.cnn_scratch import CNN


BATCH_SIZE = 64
EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42
MODEL_SAVE_PATH = Path("models/cnn_from_scratch_best.pth")


def split_data(samples):
    labels = [label for _, label in samples]

    train_val_samples, test_samples = train_test_split(
        samples,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    train_val_labels = [label for _, label in train_val_samples]

    train_samples, val_samples = train_test_split(
        train_val_samples,
        test_size=0.15 / 0.85,
        random_state=RANDOM_SEED,
        stratify=train_val_labels,
    )

    return train_samples, val_samples, test_samples


def compute_mean_std(train_samples):
    basic_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
         ]
    )

    dataset = PlantVillageDataset(train_samples, transform=basic_transform)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    mean = torch.zeros(3)
    total_pixels = 0

    for images, _ in loader:
        batch_size, _, height, width = images.shape
        mean += images.sum(dim=[0, 2, 3])
        total_pixels += batch_size * height * width

    mean = mean / total_pixels

    variance = torch.zeros(3)

    for images, _ in loader:
        variance += ((images - mean[None, :, None, None]) ** 2).sum(dim=[0, 2, 3])

    std = torch.sqrt(variance / total_pixels)

    return mean, std


def make_transforms(mean, std):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
        ]
    )

    return train_transform, eval_transform


def make_class_weights(train_samples, num_classes, device):
    labels = [label for _, label in train_samples]
    labels = torch.tensor(labels, dtype=torch.long)

    class_counts = torch.bincount(labels, minlength=num_classes).float()

    class_weights = class_counts.sum() / (num_classes * class_counts)
    class_weights = class_weights.to(device)

    return class_weights


def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def evaluate(model, data_loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    average_loss = total_loss / total
    accuracy = correct / total
    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return average_loss, accuracy, macro_f1


def save_checkpoint(model, mean, std, best_val_f1):
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "best_val_macro_f1": best_val_f1,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "random_seed": RANDOM_SEED,
        },
        MODEL_SAVE_PATH,
    )


def load_best_model(device):
    model = CNN().to(device)

    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model


def main():
    torch.manual_seed(RANDOM_SEED)

    device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Number of classes: {len(classes)}")
    print(f"Total images: {len(image_paths_and_labels)}")

    train_samples, val_samples, test_samples = split_data(image_paths_and_labels)

    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")

    mean, std = compute_mean_std(train_samples)

    print(f"Training mean: {mean}")
    print(f"Training std: {std}")

    train_transform, eval_transform = make_transforms(mean, std)

    train_dataset = PlantVillageDataset(train_samples, transform=train_transform)
    val_dataset = PlantVillageDataset(val_samples, transform=eval_transform)
    test_dataset = PlantVillageDataset(test_samples, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = CNN().to(device)

    class_weights = make_class_weights(
        train_samples=train_samples,
        num_classes=len(classes),
        device=device,
    )

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    def lr_lambda(epoch):
        if epoch < 1:
            return float(epoch + 1)

        progress = (epoch - 1) / max(1, EPOCHS - 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_f1 = 0.0
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
        )

        val_loss, val_acc, val_f1 = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
        )

        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_acc:.4f} | "
            f"Val macro-F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0

            save_checkpoint(model, mean, std, best_val_f1)

            print(f"Saved best model to {MODEL_SAVE_PATH}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training finished.")

    best_model = load_best_model(device)

    test_loss, test_acc, test_f1 = evaluate(
        best_model,
        test_loader,
        loss_fn,
        device,
    )

    print(
        f"Test loss: {test_loss:.4f} | "
        f"Test acc: {test_acc:.4f} | "
        f"Test macro-F1: {test_f1:.4f}"
    )


if __name__ == "__main__":
    main()