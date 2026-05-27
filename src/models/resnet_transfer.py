import torch
import torch.nn as nn
from torchvision import models

def build_resnet18(num_classes: int = 38, finetune_mode: str = "head_only"):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    for param in model.parameters():
        param.requires_grad = False
    
    if finetune_mode == "last_block":
        for param in model.layer4.parameters():
            param.requires_grad = True
            
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, num_classes))
    
    return model


def count_trainable_params(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")