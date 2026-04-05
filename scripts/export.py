"""
Export trained Vedu-BitNet model to .vbn binary format.

The .vbn format is a self-contained model archive readable by the
C++ and ESP32 inference engines with zero ML library dependencies.

Usage:
    python scripts/export.py

Inputs:
    checkpoints/vedu_bitnet_ternary.pt

Outputs:
    checkpoints/vedu_model.vbn
"""

import sys
import os
import struct
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from bitnet_edge import VeduBitNetCNN
from bitnet_edge.quantize import ternary_quantize

CHECKPOINT_DIR = "./checkpoints"
DATA_DIR = "./data/mnist"


def pack_ternary_weights(weight_tensor):
    """Pack ternary weights into 2-bit codes in uint32 blocks.
    Mapping: 0 -> 0b00, +1 -> 0b01, -1 -> 0b10
    16 weights per uint32."""
    w_q = ternary_quantize(weight_tensor.data)
    flat = w_q.flatten().cpu().tolist()

    code_map = {0.0: 0b00, 1.0: 0b01, -1.0: 0b10}
    codes = [code_map.get(w, 0b00) for w in flat]

    packed = []
    for i in range(0, len(codes), 16):
        chunk = codes[i:i + 16]
        while len(chunk) < 16:
            chunk.append(0b00)
        val = 0
        for j, c in enumerate(chunk):
            val |= (c << (j * 2))
        packed.append(val)

    return flat, packed


def export_model(model, filepath, test_input=None):
    """Export full model to .vbn binary format."""
    model.eval()
    model.cpu()

    layers = []

    # Conv1
    c1 = model.conv1
    layers.append({
        'type': 0,
        'out_ch': c1.weight.shape[0],
        'in_ch': c1.weight.shape[1],
        'ksize': c1.weight.shape[2],
        'stride': c1.stride,
        'padding': c1.padding,
        'weight': c1.weight,
        'bias': c1.bias,
        'norm_weight': c1.norm.weight,
        'norm_bias': c1.norm.bias,
    })

    # Conv2
    c2 = model.conv2
    layers.append({
        'type': 0,
        'out_ch': c2.weight.shape[0],
        'in_ch': c2.weight.shape[1],
        'ksize': c2.weight.shape[2],
        'stride': c2.stride,
        'padding': c2.padding,
        'weight': c2.weight,
        'bias': c2.bias,
        'norm_weight': c2.norm.weight,
        'norm_bias': c2.norm.bias,
    })

    # FC1
    f1 = model.fc1
    layers.append({
        'type': 1,
        'out_features': f1.weight.shape[0],
        'in_features': f1.weight.shape[1],
        'weight': f1.weight,
        'bias': f1.bias,
        'norm_weight': f1.norm.weight,
        'norm_bias': f1.norm.bias,
    })

    # FC2
    f2 = model.fc2
    layers.append({
        'type': 1,
        'out_features': f2.weight.shape[0],
        'in_features': f2.weight.shape[1],
        'weight': f2.weight,
        'bias': f2.bias,
        'norm_weight': f2.norm.weight,
        'norm_bias': f2.norm.bias,
    })

    with open(filepath, 'wb') as f:
        f.write(b'VBN1')
        f.write(struct.pack('<I', len(layers)))

        for layer in layers:
            f.write(struct.pack('<B', layer['type']))

            if layer['type'] == 0:
                f.write(struct.pack('<IIIII',
                    layer['out_ch'], layer['in_ch'], layer['ksize'],
                    layer['stride'], layer['padding']))
            else:
                f.write(struct.pack('<II',
                    layer['out_features'], layer['in_features']))

            flat_weights, packed = pack_ternary_weights(layer['weight'])
            f.write(struct.pack('<I', len(flat_weights)))
            f.write(struct.pack('<I', len(packed)))
            for p in packed:
                f.write(struct.pack('<I', p))

            bias_np = layer['bias'].detach().cpu().numpy().astype(np.float32)
            f.write(struct.pack('<I', len(bias_np)))
            f.write(bias_np.tobytes())

            gamma_np = layer['norm_weight'].detach().cpu().numpy().astype(np.float32)
            f.write(struct.pack('<I', len(gamma_np)))
            f.write(gamma_np.tobytes())

            beta_np = layer['norm_bias'].detach().cpu().numpy().astype(np.float32)
            f.write(struct.pack('<I', len(beta_np)))
            f.write(beta_np.tobytes())

        # Test vector
        if test_input is not None:
            f.write(b'TEST')
            inp_np = test_input.detach().cpu().numpy().astype(np.float32).flatten()
            f.write(struct.pack('<I', len(inp_np)))
            f.write(inp_np.tobytes())

            with torch.no_grad():
                out = model(test_input.unsqueeze(0).cpu())
                pred = out.argmax(dim=1).item()
                logits = out.squeeze(0).cpu().numpy().astype(np.float32)
            f.write(struct.pack('<i', pred))
            f.write(struct.pack('<I', len(logits)))
            f.write(logits.tobytes())

    file_size = os.path.getsize(filepath)
    print(f"Exported model to: {filepath}")
    print(f"File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")
    print(f"Layers: {len(layers)}")
    if test_input is not None:
        print(f"Test vector included (expected prediction: {pred})")


def main():
    model_path = os.path.join(CHECKPOINT_DIR, "vedu_bitnet_ternary.pt")
    model = VeduBitNetCNN()
    state = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded trained model from {model_path}")

    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(DATA_DIR, train=False,
                                   download=True, transform=transform)
    test_img, test_label = test_dataset[0]
    print(f"Test image label: {test_label}")

    with torch.no_grad():
        out = model(test_img.unsqueeze(0))
        pred = out.argmax(dim=1).item()
    print(f"PyTorch prediction: {pred}")

    output_path = os.path.join(CHECKPOINT_DIR, "vedu_model.vbn")
    export_model(model, output_path, test_input=test_img)


if __name__ == "__main__":
    main()
