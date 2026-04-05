"""
Model architectures for the Vedu-BitNet experiment.

Both models have identical structure:
  Conv(in_ch->16, 3x3) -> ReLU -> MaxPool
  Conv(16->32, 3x3) -> ReLU -> MaxPool
  Flatten -> Linear(32*(img_size//4)**2 -> 128) -> ReLU -> Linear(128->num_classes)

The only difference: BaselineCNN uses standard FP32 layers,
VeduBitNetCNN uses ternary-quantized layers.

Defaults match MNIST (in_channels=1, img_size=28, num_classes=10).
For CIFAR-10: in_channels=3, img_size=32, num_classes=10.
"""

import torch.nn as nn
import torch.nn.functional as F

from .layers import VeduBitConv2d, VeduBitLinear


class BaselineCNN(nn.Module):
    """Standard FP32 CNN — the control group."""

    def __init__(self, in_channels=1, img_size=28, num_classes=10):
        super().__init__()
        pool_size = img_size // 4  # two MaxPool(2) layers
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * pool_size * pool_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

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

    def __init__(self, in_channels=1, img_size=28, num_classes=10):
        super().__init__()
        pool_size = img_size // 4  # two MaxPool(2) layers
        self.conv1 = VeduBitConv2d(in_channels, 16, 3, padding=1)
        self.conv2 = VeduBitConv2d(16, 32, 3, padding=1)
        self.fc1 = VeduBitLinear(32 * pool_size * pool_size, 128)
        self.fc2 = VeduBitLinear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
