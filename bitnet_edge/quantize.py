"""
Straight-Through Estimator (STE) for ternary quantization.

FORWARD PASS:
  Snap every weight to the nearest value in {-1, 0, 1}.
  Inference becomes pure addition/subtraction — no multiplication.

BACKWARD PASS:
  Pretend quantization didn't happen.
  Gradients flow straight through to the FP32 latent weights.
"""

import torch


class TernaryQuantize(torch.autograd.Function):
    """
    Quantize weights to {-1, 0, 1} with straight-through gradient.

    Method (from BitNet b1.58 / AbsMax approach):
      1. Compute a per-tensor threshold = mean(|w|)
      2. Scale weights by 1/threshold so the "active" weights land near +/-1
      3. Round and clamp to {-1, 0, 1}

    This ensures a healthy mix of -1, 0, +1 instead of collapsing to all zeros.
    """

    @staticmethod
    def forward(ctx, weight):
        threshold = weight.abs().mean() + 1e-8
        w_scaled = weight / threshold
        w_ternary = torch.clamp(torch.round(w_scaled), -1.0, 1.0)
        ctx.save_for_backward(weight)
        return w_ternary

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ternary_quantize(weight):
    """Convenience wrapper for TernaryQuantize.apply()."""
    return TernaryQuantize.apply(weight)
