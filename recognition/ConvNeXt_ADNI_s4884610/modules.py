"""
Model components for ADNI AD vs NC classification.

Two Models:
    - BaselineCNN: standard CNN used as the comparison baseline.
    - ConvNeXt: Modernized convolutional backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    """
    4-layer basline CNN

    Input:  [B, 1, 240, 256] greyscale MRI slices
    Output: [B, 2] logits for NC / AD
    """

    def __init__(self, n_classes=2):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = self.gap(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
