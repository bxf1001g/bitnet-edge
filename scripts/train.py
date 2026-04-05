"""
Train the FP32 baseline and Vedu-BitNet ternary CNN on MNIST.

Usage:
    python scripts/train.py

Outputs:
    checkpoints/baseline_fp32.pt
    checkpoints/vedu_bitnet_ternary.pt
"""

import sys
import os
import time

# Allow imports from project root
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
EPOCHS = 15
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data/mnist"
CHECKPOINT_DIR = "./checkpoints"


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
    print(f"  Epoch {epoch}: Loss={avg_loss:.4f}  Accuracy={acc:.2f}%")
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
    print("VEDU-BITNET EXPERIMENT")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"PyTorch: {torch.__version__}")
    print()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- Load MNIST ---
    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True,
                                  transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples:     {len(test_dataset)}")
    print()

    # ==========================================================
    # Phase 1: Baseline FP32 CNN
    # ==========================================================
    print("=" * 70)
    print("PHASE 1: BASELINE FP32 CNN")
    print("=" * 70)

    baseline = BaselineCNN().to(DEVICE)
    baseline_params = count_parameters(baseline)
    print(f"  Total parameters: {baseline_params:,}")
    print(f"  FP32 memory: {baseline_params * 4:,} bytes "
          f"({baseline_params * 4 / 1024:.1f} KB)")
    print()

    optimizer_b = optim.Adam(baseline.parameters(), lr=LEARNING_RATE)

    print("  Training...")
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(baseline, train_loader, optimizer_b, epoch)
    t_baseline = time.time() - t_start

    baseline_acc = evaluate(baseline, test_loader)
    print(f"\n  Final test accuracy: {baseline_acc:.2f}%")
    print(f"  Training time: {t_baseline:.1f}s")

    baseline_path = os.path.join(CHECKPOINT_DIR, "baseline_fp32.pt")
    torch.save(baseline.state_dict(), baseline_path)
    fp32_file_size = os.path.getsize(baseline_path)
    print(f"  Saved: {baseline_path} ({fp32_file_size / 1024:.1f} KB)")
    print()

    # ==========================================================
    # Phase 2: Vedu-BitNet Ternary CNN
    # ==========================================================
    print("=" * 70)
    print("PHASE 2: VEDU-BITNET TERNARY CNN (weights ∈ {-1, 0, 1})")
    print("=" * 70)

    vedu = VeduBitNetCNN().to(DEVICE)
    vedu_params = count_parameters(vedu)
    print(f"  Total parameters: {vedu_params:,}")
    print(f"  FP32 latent memory (training): {vedu_params * 4:,} bytes")
    print(f"  INT2 deployed memory (inference): "
          f"{vedu_params * 2 // 8:,} bytes "
          f"({vedu_params * 2 / 8 / 1024:.1f} KB)")
    print()

    optimizer_v = optim.Adam(vedu.parameters(), lr=LEARNING_RATE * 3)

    print("  Training with STE (Straight-Through Estimator)...")
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(vedu, train_loader, optimizer_v, epoch)
    t_vedu = time.time() - t_start

    vedu_acc = evaluate(vedu, test_loader)
    print(f"\n  Final test accuracy: {vedu_acc:.2f}%")
    print(f"  Training time: {t_vedu:.1f}s")
    print()

    print("  Inspecting quantized weights...")
    check_ternary_weights(vedu)
    print()

    vedu_path = os.path.join(CHECKPOINT_DIR, "vedu_bitnet_ternary.pt")
    torch.save(vedu.state_dict(), vedu_path)
    vedu_file_size = os.path.getsize(vedu_path)
    print(f"  Saved: {vedu_path} ({vedu_file_size / 1024:.1f} KB)")

    # ==========================================================
    # Phase 3: Comparison
    # ==========================================================
    print()
    print("=" * 70)
    print("PHASE 3: HEAD-TO-HEAD COMPARISON")
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
└─────────────────────────────┴──────────────┴──────────────┘
""")

    # Scaling projections
    print("  SCALING PROJECTIONS (same architecture, larger models):")
    print("  ─────────────────────────────────────────────────────")
    for name, param_count in [("This model", weight_params),
                               ("1M params", 1_000_000),
                               ("10M params", 10_000_000),
                               ("100M params", 100_000_000),
                               ("1B params", 1_000_000_000),
                               ("7B params", 7_000_000_000)]:
        fp32_mb = param_count * 4 / (1024 ** 2)
        int2_mb = param_count * 2 / 8 / (1024 ** 2)
        print(f"    {name:>12s}: FP32={fp32_mb:>10,.1f} MB  →  "
              f"INT2={int2_mb:>10,.1f} MB  "
              f"(saves {fp32_mb - int2_mb:>10,.1f} MB)")
    print()
    print("  Experiment complete.")


if __name__ == "__main__":
    main()
