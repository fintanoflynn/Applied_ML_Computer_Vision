from pathlib import Path
from dataclasses import dataclass
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".JPG", ".JPEG"}

@dataclass
class Sample:
    path: Path
    label: int

class PlantVillageDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.classes, self.samples = self._scan()
    
    def _scan(self):
        classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        
        class_to_idx = {cls : idx for idx, cls in enumerate(classes)}
        
        samples = []
        for cls in classes:
            cls_folder = self.root / cls
            for file in cls_folder.iterdir():
                if file.suffix in IMAGE_EXTENSIONS:
                    samples.append(Sample(path=file, label=class_to_idx[cls]))
    
        return classes, samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, sample.label
    
    def class_counts(self):
        counts = np.zeros(len(self.classes), dtype=np.int64)
        for sample in self.samples:
            counts[sample.label] += 1
        return counts
    
    @property
    def labels(self):
        return np.array([sample.label for sample in self.samples])
    
def stratified_splits(root, val_size=0.15, test_size=0.15, seed=42, train_transform=None, eval_transform=None):
        
    full_dataset = PlantVillageDataset(root, transform=None)
        
    from sklearn.model_selection import train_test_split
    
    indices = np.arange(len(full_dataset))
    labels = full_dataset.labels
    
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=val_size + test_size,
        stratify=labels,
        random_state=seed
    )
    
    temp_labels = labels[temp_idx]
    relative_test_size = test_size / (test_size + val_size)
    
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test_size,
        stratify=temp_labels,
        random_state=seed
    )
    
    train_ds = PlantVillageDataset(
        root,
        transform=train_transform
    )
    train_ds.samples = [full_dataset.samples[i] for i in train_idx]
    
    val_ds = PlantVillageDataset(
        root,
        transform=eval_transform
    )
    val_ds.samples = [full_dataset.samples[i] for i in val_idx]
    
    test_ds = PlantVillageDataset(
        root,
        transform = eval_transform
    )
    test_ds.samples = [full_dataset.samples[i] for i in test_idx]
    
    return train_ds, val_ds, test_ds