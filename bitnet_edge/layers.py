"""
Custom ternary layers for Vedu-BitNet.

These layers store FP32 latent weights for training, but quantize them
to {-1, 0, 1} in every forward pass via the Straight-Through Estimator.

During inference on hardware, the FP32 weights are discarded — only the
2-bit ternary codes are kept.

Based on Microsoft's BitLinear implementation: normalize input, quantize
weights, matmul. No output rescaling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantize import ternary_quantize


class VeduBitConv2d(nn.Module):
    """Conv2d with ternary-quantized weights via STE.

    Pipeline per forward call:
      1. GroupNorm(1, in_channels) on input — stabilizes activations
      2. Quantize weights to {-1, 0, 1}
      3. Convolution — on hardware this is pure add/sub
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.norm = nn.GroupNorm(1, in_channels)

    def forward(self, x):
        x = self.norm(x)
        w_ternary = ternary_quantize(self.weight)
        return F.conv2d(x, w_ternary, self.bias,
                        stride=self.stride, padding=self.padding)


class VeduBitLinear(nn.Module):
    """Linear layer with ternary-quantized weights via STE.

    Pipeline per forward call:
      1. LayerNorm(in_features) on input
      2. Quantize weights to {-1, 0, 1}
      3. Linear matmul — on hardware this is pure add/sub
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.norm = nn.LayerNorm(in_features)

    def forward(self, x):
        x = self.norm(x)
        w_ternary = ternary_quantize(self.weight)
        return F.linear(x, w_ternary, self.bias)
