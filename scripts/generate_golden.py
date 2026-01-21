#!/usr/bin/env python3
"""
Generate golden reference data for NeuroGrid validation.

This script runs PyTorch Llama inference and saves intermediate outputs
for validation against our CUDA implementation.

Usage:
    python generate_golden.py --model llama-7b --layer 0 --output tests/golden/

Requirements:
    pip install torch transformers safetensors
"""

import argparse
import os
import struct
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig


def save_tensor(tensor: torch.Tensor, path: str):
    """Save tensor as raw FP32 binary file."""
    # Convert to FP32 and flatten
    data = tensor.detach().float().cpu().numpy().flatten()
    with open(path, 'wb') as f:
        for val in data:
            f.write(struct.pack('<f', val))
    print(f"Saved {path}: shape={list(tensor.shape)}, dtype={tensor.dtype}")


def save_tensor_int8(tensor: torch.Tensor, path: str):
    """Save INT8 tensor as raw binary."""
    data = tensor.detach().cpu().numpy().flatten().astype('int8')
    with open(path, 'wb') as f:
        f.write(data.tobytes())
    print(f"Saved {path}: shape={list(tensor.shape)}, dtype=int8")


def compute_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm as used in Llama."""
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return x_normed * weight


def compute_silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU activation: x * sigmoid(x)"""
    return x * torch.sigmoid(x)


def compute_rope(x: torch.Tensor, positions: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Rotary Position Embeddings."""
    # x: [batch, seq, num_heads, head_dim]
    batch, seq_len, num_heads, _ = x.shape

    # Create frequency bands
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=x.device).float() / head_dim))

    # Compute sinusoids
    t = positions.float().unsqueeze(-1)  # [batch, seq, 1]
    freqs = t * inv_freq.unsqueeze(0).unsqueeze(0)  # [batch, seq, head_dim//2]

    cos = torch.cos(freqs).unsqueeze(2)  # [batch, seq, 1, head_dim//2]
    sin = torch.sin(freqs).unsqueeze(2)  # [batch, seq, 1, head_dim//2]

    # Apply rotation
    x1, x2 = x[..., :head_dim//2], x[..., head_dim//2:]
    x_rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)

    return x_rotated


def generate_layer_golden(model, layer_idx: int, output_dir: Path, device: str = "cuda"):
    """Generate golden data for a single transformer layer."""
    output_dir.mkdir(parents=True, exist_ok=True)

    config = model.config
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = hidden_size // num_heads

    print(f"\nGenerating golden data for layer {layer_idx}")
    print(f"  hidden_size={hidden_size}, num_heads={num_heads}, head_dim={head_dim}")

    # Get the layer
    layer = model.model.layers[layer_idx]

    # Create random input
    batch_size = 1
    seq_len = 4

    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float16)
    positions = torch.arange(seq_len, device=device).unsqueeze(0)

    # Save input
    save_tensor(hidden_states, output_dir / "layer_0_input.bin")

    # Get rotary embeddings from model
    # Newer transformers versions require position_embeddings instead of position_ids
    rotary_emb = model.model.rotary_emb
    position_embeddings = rotary_emb(hidden_states, positions)

    # Run layer forward
    with torch.no_grad():
        # Get output - handle different transformers versions
        try:
            # Newer transformers (>=4.45) use position_embeddings
            output = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                use_cache=False
            )
            if isinstance(output, tuple):
                output = output[0]
        except TypeError:
            # Older transformers use position_ids
            output = layer(hidden_states, position_ids=positions, use_cache=False)
            if isinstance(output, tuple):
                output = output[0]

    # Save output
    save_tensor(output, output_dir / "layer_0_output.bin")

    # Save intermediate tensors for detailed validation
    with torch.no_grad():
        residual = hidden_states

        # 1. Input norm
        normed = layer.input_layernorm(hidden_states)
        save_tensor(normed, output_dir / "after_input_norm.bin")

        # 2. Q, K, V projections
        q = layer.self_attn.q_proj(normed)
        k = layer.self_attn.k_proj(normed)
        v = layer.self_attn.v_proj(normed)

        save_tensor(q, output_dir / "q_proj_output.bin")
        save_tensor(k, output_dir / "k_proj_output.bin")
        save_tensor(v, output_dir / "v_proj_output.bin")

    # Save layer weights
    weights_dir = output_dir / "layer_0_weights"
    weights_dir.mkdir(exist_ok=True)

    save_tensor(layer.input_layernorm.weight, weights_dir / "input_layernorm.bin")
    save_tensor(layer.post_attention_layernorm.weight, weights_dir / "post_attn_layernorm.bin")
    save_tensor(layer.self_attn.q_proj.weight, weights_dir / "q_proj.bin")
    save_tensor(layer.self_attn.k_proj.weight, weights_dir / "k_proj.bin")
    save_tensor(layer.self_attn.v_proj.weight, weights_dir / "v_proj.bin")
    save_tensor(layer.self_attn.o_proj.weight, weights_dir / "o_proj.bin")
    save_tensor(layer.mlp.gate_proj.weight, weights_dir / "gate_proj.bin")
    save_tensor(layer.mlp.up_proj.weight, weights_dir / "up_proj.bin")
    save_tensor(layer.mlp.down_proj.weight, weights_dir / "down_proj.bin")

    print(f"\nGolden data saved to {output_dir}")


def generate_kernel_golden(output_dir: Path, device: str = "cuda"):
    """Generate golden data for individual kernel tests."""
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)

    # RMSNorm test data
    print("\nGenerating RMSNorm golden data...")
    hidden_dim = 4096
    x = torch.randn(1, 1, hidden_dim, device=device, dtype=torch.float16)
    weight = torch.ones(hidden_dim, device=device, dtype=torch.float16)
    eps = 1e-6

    # Compute RMSNorm
    variance = x.pow(2).mean(-1, keepdim=True)
    y = x * torch.rsqrt(variance + eps) * weight

    save_tensor(x, output_dir / "rmsnorm_input.bin")
    save_tensor(weight, output_dir / "rmsnorm_weight.bin")
    save_tensor(y, output_dir / "rmsnorm_output.bin")

    # SiLU test data
    print("Generating SiLU golden data...")
    x_silu = torch.randn(11008, device=device, dtype=torch.float16)
    y_silu = x_silu * torch.sigmoid(x_silu)

    save_tensor(x_silu, output_dir / "silu_input.bin")
    save_tensor(y_silu, output_dir / "silu_output.bin")

    # GEMM test data
    print("Generating GEMM golden data...")
    M, K, N = 32, 128, 64
    a = torch.randn(M, K, device=device, dtype=torch.float16)
    b = torch.randn(K, N, device=device, dtype=torch.float16)
    c = torch.matmul(a, b)

    save_tensor(a, output_dir / "gemm_a.bin")
    save_tensor(b, output_dir / "gemm_b.bin")
    save_tensor(c, output_dir / "gemm_c.bin")

    print(f"\nKernel golden data saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate golden reference data")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf",
                       help="Model name or path")
    parser.add_argument("--layer", type=int, default=0,
                       help="Layer index to generate data for")
    parser.add_argument("--output", type=str, default="tests/golden/",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda or cpu)")
    parser.add_argument("--kernels-only", action="store_true",
                       help="Only generate kernel test data (no model needed)")
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.kernels_only:
        generate_kernel_golden(output_dir, args.device)
        return

    print(f"Loading model: {args.model}")

    # Check if model exists locally or needs auth
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map=args.device,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("\nTo use Llama models, you need:")
        print("1. Accept the license at https://huggingface.co/meta-llama/Llama-2-7b-hf")
        print("2. Login with: huggingface-cli login")
        print("\nAlternatively, use --kernels-only for basic kernel tests")
        return

    model.eval()

    generate_layer_golden(model, args.layer, output_dir, args.device)
    generate_kernel_golden(output_dir / "kernels", args.device)


if __name__ == "__main__":
    main()
