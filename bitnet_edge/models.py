"""
Model architectures for the Vedu-BitNet experiment.

Both models have identical structure:
  Conv(1->16, 3x3) -> ReLU -> MaxPool
  Conv(16->32, 3x3) -> ReLU -> MaxPool
  Flatten -> Linear(1568->128) -> ReLU -> Linear(128->10)

The only difference: BaselineCNN uses standard FP32 layers,
VeduBitNetCNN uses ternary-quantized layers.
"""

import torch.nn as nn
import torch.nn.functional as F

from .layers import VeduBitConv2d, VeduBitLinear


class BaselineCNN(nn.Module):
    """Standard FP32 CNN — the control group."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class VeduBitNetCNN(nn.Module):
    """Ternary-weight CNN — the experimental model."""

    def __init__(self):
        super().__init__()
        self.conv1 = VeduBitConv2d(1, 16, 3, padding=1)
        self.conv2 = VeduBitConv2d(16, 32, 3, padding=1)
        self.fc1 = VeduBitLinear(32 * 7 * 7, 128)
        self.fc2 = VeduBitLinear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
