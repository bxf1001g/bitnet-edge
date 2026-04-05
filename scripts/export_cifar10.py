"""
Export trained CIFAR-10 Vedu-BitNet model to .vbn binary format.

Usage:
    python scripts/export_cifar10.py

Inputs:
    checkpoints/cifar10_vedu_bitnet_ternary.pt

Outputs:
    checkpoints/cifar10_vedu_model.vbn
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from torchvision import datasets, transforms

from bitnet_edge import VeduBitNetCNN
from scripts.export import export_model

# ============================================================
# Configuration
# ============================================================
CHECKPOINT_DIR = "./checkpoints"
DATA_DIR = "./data/cifar10"

IN_CHANNELS = 3
IMG_SIZE = 32
NUM_CLASSES = 10

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def main():
    model_path = os.path.join(CHECKPOINT_DIR, "cifar10_vedu_bitnet_ternary.pt")
    model = VeduBitNetCNN(in_channels=IN_CHANNELS, img_size=IMG_SIZE,
                          num_classes=NUM_CLASSES)
    state = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded trained model from {model_path}")

    # Load a test image from CIFAR-10
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    test_dataset = datasets.CIFAR10(DATA_DIR, train=False,
                                     download=True, transform=test_transform)
    test_img, test_label = test_dataset[0]
    print(f"Test image label: {test_label} ({CIFAR10_CLASSES[test_label]})")

    with torch.no_grad():
        out = model(test_img.unsqueeze(0))
        pred = out.argmax(dim=1).item()
    print(f"PyTorch prediction: {pred} ({CIFAR10_CLASSES[pred]})")

    output_path = os.path.join(CHECKPOINT_DIR, "cifar10_vedu_model.vbn")
    export_model(model, output_path, test_input=test_img)


if __name__ == "__main__":
    main()
