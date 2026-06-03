import torch
import torch.nn as nn
import shutil
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from src.data.dataset import stratified_splits
from src.data.transforms import build_train_transform, build_eval_transform
from src.models.resnet_transfer import build_resnet18
from src.utils.seed import set_seed


def compute_class_weights(dataset, device):
    counts = dataset.class_counts()
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    
    return torch.tensor(weights, dtype=torch.float32).to(device)


def train(data_root, num_epochs=30, lr=1e-4, batch_size=64, dropout=0.3, finetune_mode="head_only", seed=42):
    
    import os
    os.makedirs('checkpoints', exist_ok=True)
    
    set_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_ds, val_ds, test_ds = stratified_splits(
        data_root,
        train_transform=build_train_transform(),
        eval_transform=build_eval_transform()
    )
    
    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    model = build_resnet18(num_classes=len(train_ds.classes), finetune_mode=finetune_mode, dropout=dropout)
    model = model.to(device)
    
    class_weights = compute_class_weights(train_ds, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=num_epochs
    )
    
    best_val_macro_f1 = 0.0
    patience_counter = 0
    patience = 5
    
    for epoch in range(1, num_epochs + 1):
        
        # ── Training phase ──────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        total_batches = len(train_loader)
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimiser.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch} | batch {batch_idx+1}/{total_batches} | "
                      f"loss={loss.item():.4f}")
            
            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += images.size(0)
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # ── Validation phase ────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += images.size(0)
                
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        val_macro_f1 = f1_score(val_true, val_preds, average="macro")
        
        scheduler.step()
        
        print(f"Epoch {epoch:02d}/{num_epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"val_macro_f1={val_macro_f1:.4f}"
              )
        
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimiser_state": optimiser.state_dict(),
                "val_acc": val_acc,
                "val_macro_f1": val_macro_f1,
                "classes": train_ds.classes
            }, "checkpoints/resnet18_best.pt")
            
            drive_path = "/content/drive/MyDrive/resnet18_best.pt"
            if os.path.exists("/content/drive/MyDrive"):
                shutil.copy("checkpoints/resnet18_best.pt", drive_path)
                print(f"Checkpoint saved to Drive folder")
            print(f"New best validation macro f1={val_macro_f1:.4f} - model saved")

        else:
            patience_counter += 1
            print(f"No improvement seen ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
            
    print(f"\nTraining complete. Best validation macro f1={best_val_macro_f1:.4f}")
    return best_val_macro_f1