import torch
import sys
from torch.utils.data import DataLoader
from src.data.dataset import PlantVillageDataset, stratified_splits
from src.data.transforms import build_eval_transform
from src.models.resnet_transfer import build_resnet18
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def evaluate(checkpoint_path, data_root):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint['classes']
    num_classes = len(classes)

    # Load model
    model = build_resnet18(num_classes=num_classes, finetune_mode='head_only')
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()

    # Load test set
    _, _, test_ds = stratified_splits(
        data_root,
        eval_transform=build_eval_transform()
    )

    
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall accuracy
    accuracy = (all_preds == all_labels).mean()
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Per class breakdown
    print("\nPer-class breakdown:")
    print(classification_report(
        all_labels, all_preds,
        target_names=classes,
        digits=3
    ))

    # Class counts in test set
    print("\nTest set class distribution:")
    unique, counts = np.unique(all_labels, return_counts=True)
    for idx, count in zip(unique, counts):
        print(f"  {classes[idx]:<50} {count} images")


if __name__ == "__main__":
    evaluate('checkpoints/resnet18_best.pt', 'data/raw')