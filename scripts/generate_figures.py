"""
Generate publication-quality figures for the bitnet-edge paper.
Outputs PNG files to docs/figures/
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = "./docs/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Use a clean style
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

# Color scheme
C_FP32 = '#2196F3'      # blue
C_TERNARY = '#FF9800'   # orange
C_MNIST = '#4CAF50'     # green
C_CIFAR = '#E91E63'     # pink

# ============================================================
# 1. MNIST Training Curves
# ============================================================
mnist_baseline_acc = [
    61.51, 95.49, 96.66, 97.41, 97.57, 97.74, 97.91, 98.08,
    98.19, 98.29, 98.36, 98.48, 98.51, 98.59, 98.64
]
mnist_ternary_acc = [
    55.20, 91.12, 93.75, 95.22, 95.99, 96.58, 96.91, 97.13,
    97.44, 97.62, 97.73, 97.89, 97.98, 98.12, 98.25
]

fig, ax = plt.subplots(figsize=(8, 5))
epochs = range(1, 16)
ax.plot(epochs, mnist_baseline_acc, '-o', color=C_FP32, label='FP32 Baseline (99.02%)', markersize=5)
ax.plot(epochs, mnist_ternary_acc, '-s', color=C_TERNARY, label='Ternary Vedu-BitNet (98.79%)', markersize=5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Training Accuracy (%)')
ax.set_title('MNIST Training Progress')
ax.legend(loc='lower right')
ax.set_ylim(50, 100)
ax.grid(True, alpha=0.3)
ax.axhline(y=99.02, color=C_FP32, linestyle='--', alpha=0.4)
ax.axhline(y=98.79, color=C_TERNARY, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/mnist_training_curves.png")
plt.close()
print("1/6 mnist_training_curves.png")

# ============================================================
# 2. CIFAR-10 Training Curves
# ============================================================
cifar_baseline_acc = [
    40.71, 51.55, 56.61, 60.17, 61.92, 64.00, 65.28, 66.43,
    67.45, 67.98, 68.93, 69.95, 70.42, 70.89, 71.58,
    71.85, 72.46, 72.76, 73.00, 73.45, 73.62, 73.99,
    74.00, 74.28, 74.52, 74.65, 74.82, 74.91, 74.87, 75.05
]
cifar_ternary_acc = [
    31.33, 41.45, 45.66, 49.76, 50.68, 53.59, 55.43, 57.01,
    59.09, 60.24, 61.26, 60.92, 62.04, 61.97, 63.26,
    63.80, 64.48, 64.84, 65.75, 65.96, 66.32, 66.51,
    66.99, 67.39, 67.74, 68.28, 68.21, 68.79, 69.16, 69.76
]

fig, ax = plt.subplots(figsize=(8, 5))
epochs = range(1, 31)
ax.plot(epochs, cifar_baseline_acc, '-o', color=C_FP32, label='FP32 Baseline (75.52%)', markersize=4)
ax.plot(epochs, cifar_ternary_acc, '-s', color=C_TERNARY, label='Ternary Vedu-BitNet (69.02%)', markersize=4)
ax.set_xlabel('Epoch')
ax.set_ylabel('Training Accuracy (%)')
ax.set_title('CIFAR-10 Training Progress')
ax.legend(loc='lower right')
ax.set_ylim(25, 80)
ax.grid(True, alpha=0.3)
ax.axhline(y=75.52, color=C_FP32, linestyle='--', alpha=0.4, label='_')
ax.axhline(y=69.02, color=C_TERNARY, linestyle='--', alpha=0.4, label='_')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/cifar10_training_curves.png")
plt.close()
print("2/6 cifar10_training_curves.png")

# ============================================================
# 3. Accuracy Comparison Bar Chart
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(2)
width = 0.3

bars1 = ax.bar(x - width/2, [99.02, 75.52], width, label='FP32 Baseline', color=C_FP32, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, [98.79, 69.02], width, label='Ternary (Vedu-BitNet)', color=C_TERNARY, edgecolor='black', linewidth=0.5)

ax.set_ylabel('Test Accuracy (%)')
ax.set_title('FP32 vs Ternary Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(['MNIST\n(1×28×28)', 'CIFAR-10\n(3×32×32)'])
ax.legend()
ax.set_ylim(0, 110)
ax.grid(True, alpha=0.2, axis='y')

# Add value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add drop annotations
ax.annotate('-0.23 pp', xy=(0.15, 98.79), fontsize=9, color='red', fontweight='bold')
ax.annotate('-6.50 pp', xy=(1.15, 69.02), fontsize=9, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/accuracy_comparison.png")
plt.close()
print("3/6 accuracy_comparison.png")

# ============================================================
# 4. Weight Distribution Pie Charts
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# MNIST
mnist_dist = [72000, 40000, 95000]
colors = ['#E53935', '#9E9E9E', '#43A047']
labels = ['-1', '0', '+1']
explode = (0.03, 0.03, 0.03)

wedges1, texts1, autotexts1 = ax1.pie(mnist_dist, labels=labels, colors=colors,
    autopct='%1.1f%%', explode=explode, startangle=90,
    textprops={'fontsize': 11})
for t in autotexts1:
    t.set_fontweight('bold')
ax1.set_title('MNIST Weight Distribution', fontsize=13)

# CIFAR-10
cifar_dist = [84760, 93329, 92570]
wedges2, texts2, autotexts2 = ax2.pie(cifar_dist, labels=labels, colors=colors,
    autopct='%1.1f%%', explode=explode, startangle=90,
    textprops={'fontsize': 11})
for t in autotexts2:
    t.set_fontweight('bold')
ax2.set_title('CIFAR-10 Weight Distribution', fontsize=13)

plt.suptitle('Ternary Weight Distribution After Training', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/weight_distribution.png", bbox_inches='tight')
plt.close()
print("4/6 weight_distribution.png")

# ============================================================
# 5. Memory Compression Chart
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))

categories = ['MNIST\nWeights', 'CIFAR-10\nWeights']
fp32_mem = [203, 1074]
ternary_mem = [12.7, 67]

x = np.arange(len(categories))
width = 0.3

bars1 = ax.bar(x - width/2, fp32_mem, width, label='FP32 (32-bit)', color=C_FP32, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, ternary_mem, width, label='Ternary (2-bit)', color=C_TERNARY, edgecolor='black', linewidth=0.5)

ax.set_ylabel('Weight Memory (KB)')
ax.set_title('Memory Compression: 93.75% Reduction')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.grid(True, alpha=0.2, axis='y')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 15,
            f'{bar.get_height():.0f} KB', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 15,
            f'{bar.get_height():.1f} KB', ha='center', va='bottom', fontsize=10)

# Add 16x arrow
ax.annotate('16×\nsmaller', xy=(0.15, 50), fontsize=11, color='#D32F2F',
            fontweight='bold', ha='center')
ax.annotate('16×\nsmaller', xy=(1.15, 200), fontsize=11, color='#D32F2F',
            fontweight='bold', ha='center')

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/memory_compression.png")
plt.close()
print("5/6 memory_compression.png")

# ============================================================
# 6. ESP32 Latency Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))

platforms = ['Desktop C++\n(x86-64)', 'ESP32-S3\n(240 MHz)']
mnist_lat = [4.97, 194.9]
cifar_lat = [3.0, 333.3]

x = np.arange(len(platforms))
width = 0.3

bars1 = ax.bar(x - width/2, mnist_lat, width, label='MNIST', color=C_MNIST, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, cifar_lat, width, label='CIFAR-10', color=C_CIFAR, edgecolor='black', linewidth=0.5)

ax.set_ylabel('Inference Latency (ms)')
ax.set_title('Inference Latency by Platform')
ax.set_xticks(x)
ax.set_xticklabels(platforms)
ax.legend()
ax.grid(True, alpha=0.2, axis='y')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
            f'{bar.get_height():.1f} ms', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
            f'{bar.get_height():.1f} ms', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/inference_latency.png")
plt.close()
print("6/6 inference_latency.png")

print(f"\nAll figures saved to {OUT_DIR}/")
