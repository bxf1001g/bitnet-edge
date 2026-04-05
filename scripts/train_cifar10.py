"""
Train the FP32 baseline and Vedu-BitNet ternary CNN on CIFAR-10.

Usage:
    python scripts/train_cifar10.py

Outputs:
    checkpoints/cifar10_baseline_fp32.pt
    checkpoints/cifar10_vedu_bitnet_ternary.pt
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from bitnet_edge import BaselineCNN, VeduBitNetCNN
from bitnet_edge.quantize import ternary_quantize

# ============================================================
# Configuration
# ============================================================
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data/cifar10"
CHECKPOINT_DIR = "./checkpoints"

# CIFAR-10 model parameters
IN_CHANNELS = 3
IMG_SIZE = 32
NUM_CLASSES = 10
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ============================================================
# Training & Evaluation
# ============================================================

def train_one_epoch(model, loader, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    acc = 100.0 * correct / total
    avg_loss = total_loss / len(loader)
    print(f"  Epoch {epoch:>2d}: Loss={avg_loss:.4f}  Accuracy={acc:.2f}%")
    return acc


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return 100.0 * correct / total


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_weight_parameters(model):
    total = 0
    for name, p in model.named_parameters():
        if 'bias' not in name and p.requires_grad:
            total += p.numel()
    return total


def check_ternary_weights(model):
    stats = {-1: 0, 0: 0, 1: 0}
    total = 0
    for name, p in model.named_parameters():
        if 'bias' not in name:
            w_q = ternary_quantize(p.data)
            for val in [-1, 0, 1]:
                stats[val] += (w_q == val).sum().item()
            total += w_q.numel()

    print(f"  Weight distribution after quantization:")
    for val in [-1, 0, 1]:
        pct = 100.0 * stats[val] / total
        bar = "█" * int(pct / 2)
        print(f"    {val:+d}: {stats[val]:>7d} ({pct:5.1f}%) {bar}")
    print(f"    Total weights: {total}")
    sparsity = 100.0 * stats[0] / total
    print(f"    Zero-weights (FREE sparsity): {sparsity:.1f}%")
    return stats


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("VEDU-BITNET CIFAR-10 EXPERIMENT")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Image: {IN_CHANNELS}x{IMG_SIZE}x{IMG_SIZE}  Classes: {NUM_CLASSES}")
    print()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- Load CIFAR-10 ---
    print("Loading CIFAR-10 dataset...")

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = datasets.CIFAR10(DATA_DIR, train=True, download=True,
                                     transform=train_transform)
    test_dataset = datasets.CIFAR10(DATA_DIR, train=False, download=True,
                                    transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=2, pin_memory=True)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples:     {len(test_dataset)}")
    print()

    # ==========================================================
    # Phase 1: Baseline FP32 CNN
    # ==========================================================
    print("=" * 70)
    print("PHASE 1: BASELINE FP32 CNN (CIFAR-10)")
    print("=" * 70)

    baseline = BaselineCNN(in_channels=IN_CHANNELS, img_size=IMG_SIZE,
                           num_classes=NUM_CLASSES).to(DEVICE)
    baseline_params = count_parameters(baseline)
    print(f"  Total parameters: {baseline_params:,}")
    print(f"  FP32 memory: {baseline_params * 4:,} bytes "
          f"({baseline_params * 4 / 1024:.1f} KB)")
    print()

    optimizer_b = optim.Adam(baseline.parameters(), lr=LEARNING_RATE)
    scheduler_b = optim.lr_scheduler.CosineAnnealingLR(optimizer_b, T_max=EPOCHS)

    print("  Training...")
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(baseline, train_loader, optimizer_b, epoch)
        scheduler_b.step()
    t_baseline = time.time() - t_start

    baseline_acc = evaluate(baseline, test_loader)
    print(f"\n  Final test accuracy: {baseline_acc:.2f}%")
    print(f"  Training time: {t_baseline:.1f}s")

    baseline_path = os.path.join(CHECKPOINT_DIR, "cifar10_baseline_fp32.pt")
    torch.save(baseline.state_dict(), baseline_path)
    fp32_file_size = os.path.getsize(baseline_path)
    print(f"  Saved: {baseline_path} ({fp32_file_size / 1024:.1f} KB)")
    print()

    # ==========================================================
    # Phase 2: Vedu-BitNet Ternary CNN
    # ==========================================================
    print("=" * 70)
    print("PHASE 2: VEDU-BITNET TERNARY CNN (CIFAR-10)")
    print("=" * 70)

    vedu = VeduBitNetCNN(in_channels=IN_CHANNELS, img_size=IMG_SIZE,
                         num_classes=NUM_CLASSES).to(DEVICE)
    vedu_params = count_parameters(vedu)
    print(f"  Total parameters: {vedu_params:,}")
    print(f"  FP32 latent memory (training): {vedu_params * 4:,} bytes")
    print(f"  INT2 deployed memory (inference): "
          f"{vedu_params * 2 // 8:,} bytes "
          f"({vedu_params * 2 / 8 / 1024:.1f} KB)")
    print()

    optimizer_v = optim.Adam(vedu.parameters(), lr=LEARNING_RATE * 3)
    scheduler_v = optim.lr_scheduler.CosineAnnealingLR(optimizer_v, T_max=EPOCHS)

    print("  Training with STE (Straight-Through Estimator)...")
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(vedu, train_loader, optimizer_v, epoch)
        scheduler_v.step()
    t_vedu = time.time() - t_start

    vedu_acc = evaluate(vedu, test_loader)
    print(f"\n  Final test accuracy: {vedu_acc:.2f}%")
    print(f"  Training time: {t_vedu:.1f}s")
    print()

    print("  Inspecting quantized weights...")
    check_ternary_weights(vedu)
    print()

    vedu_path = os.path.join(CHECKPOINT_DIR, "cifar10_vedu_bitnet_ternary.pt")
    torch.save(vedu.state_dict(), vedu_path)
    vedu_file_size = os.path.getsize(vedu_path)
    print(f"  Saved: {vedu_path} ({vedu_file_size / 1024:.1f} KB)")

    # ==========================================================
    # Phase 3: Comparison
    # ==========================================================
    print()
    print("=" * 70)
    print("PHASE 3: HEAD-TO-HEAD COMPARISON (CIFAR-10)")
    print("=" * 70)

    weight_params = count_weight_parameters(baseline)
    fp32_weight_bytes = weight_params * 4
    int2_weight_bytes = (weight_params * 2) // 8
    compression = (1 - int2_weight_bytes / fp32_weight_bytes) * 100

    print(f"""
┌─────────────────────────────┬──────────────┬──────────────┐
│ Metric                      │ FP32 Baseline│ Vedu-BitNet  │
├─────────────────────────────┼──────────────┼──────────────┤
│ Test Accuracy               │ {baseline_acc:>10.2f}% │ {vedu_acc:>10.2f}% │
│ Accuracy Drop               │          --- │ {baseline_acc - vedu_acc:>+10.2f}% │
├─────────────────────────────┼──────────────┼──────────────┤
│ Bits per weight             │           32 │            2 │
│ Weight memory               │ {fp32_weight_bytes:>9,} B │ {int2_weight_bytes:>9,} B │
│ Memory compression          │          --- │     {compression:.1f}%  │
├─────────────────────────────┼──────────────┼──────────────┤
│ Training time               │ {t_baseline:>10.1f}s │ {t_vedu:>10.1f}s │
│ Multiplies needed (infer)   │  {weight_params:>10,} │            0 │
│ Operations (inference)      │    MUL + ADD │   ADD/SUB    │
├─────────────────────────────┼──────────────┼──────────────┤
│ Input                       │ 3x32x32 RGB │ 3x32x32 RGB │
│ Classes                     │ {NUM_CLASSES:>12d} │ {NUM_CLASSES:>12d} │
└─────────────────────────────┴──────────────┴──────────────┘
""")

    print("  CIFAR-10 CLASSES:")
    for i, name in enumerate(CIFAR10_CLASSES):
        print(f"    {i}: {name}")
    print()


if __name__ == "__main__":
    main()
