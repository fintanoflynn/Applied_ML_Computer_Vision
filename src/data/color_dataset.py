""" Building the colour dataset """

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

class PlantVillageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

root = Path("data/raw/plantvillage/plantvillage dataset/color")

classes = sorted(folder.name for folder in root.iterdir() if folder.is_dir())

class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}

image_paths_and_labels = []

for class_name in classes:
    class_folder = root / class_name
    label = class_to_idx[class_name]

    for image_path in class_folder.iterdir():
        if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            image_paths_and_labels.append((image_path, label))

print(f"Found {len(image_paths_and_labels)} images")
print(image_paths_and_labels[:3])