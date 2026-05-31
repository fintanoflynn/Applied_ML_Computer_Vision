from __future__ import annotations
import torch
from torch import nn

class CNN(nn.Module):
    
    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def __init__(self):
        super(CNN, self).__init__()
        
        # 4 convolutional blocks with increasing number of filters
        self.feature_extractor = nn.Sequential(
            self._conv_block(3, 32),
            self._conv_block(32, 64),
            self._conv_block(64, 128),
            self._conv_block(128, 256)
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(256, 38),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(inputs)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        output = self.classifier(x)
        return output
        