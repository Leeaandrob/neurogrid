#!/usr/bin/env python3
"""Debug: compare NeuroGrid hidden states against golden data layer by layer."""
import json
import numpy as np
import struct
import sys
import os

golden_dir = sys.argv[1] if len(sys.argv) > 1 else "./tests/golden/lfm2"

# Load golden data
with open(os.path.join(golden_dir, "input_tokens.json")) as f:
    token_data = json.load(f)

with open(os.path.join(golden_dir, "model_info.json")) as f:
    model_info = json.load(f)

print(f"Model: {model_info['model_type']}, layers={model_info['num_hidden_layers']}")
print(f"Layer types: {model_info['layer_types']}")
print(f"Prompt tokens: {token_data['num_tokens']} tokens")
print()

# Load and display golden hidden state norms
print("Golden hidden state norms (last token, per layer):")
for i in range(model_info['num_hidden_layers'] + 1):
    path = os.path.join(golden_dir, f"hidden_state_layer_{i}.bin")
    if os.path.exists(path):
        hs = np.fromfile(path, dtype=np.float32)
        norm = np.linalg.norm(hs)
        mean = np.mean(hs)
        std = np.std(hs)
        # Show first 8 values
        first8 = hs[:8]
        layer_type = "embed" if i == 0 else model_info['layer_types'][i-1] if i <= len(model_info['layer_types']) else "?"
        print(f"  Layer {i:2d} ({layer_type:15s}): norm={norm:10.4f}, mean={mean:+.6f}, std={std:.6f}, first8={np.array2string(first8, precision=4, separator=', ')}")

# Load logits
logits_path = os.path.join(golden_dir, "logits.bin")
if os.path.exists(logits_path):
    logits = np.fromfile(logits_path, dtype=np.float32)
    top5_idx = np.argsort(logits)[-5:][::-1]
    print(f"\nGolden logits: top-5 indices = {top5_idx.tolist()}")
    print(f"  Top-5 values: {logits[top5_idx].tolist()}")
    print(f"  Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
