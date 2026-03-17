#!/usr/bin/env python3
"""
Generate golden test data for LFM2.5-1.2B-Thinking model.

This script loads the model using HuggingFace transformers and generates:
1. Tokenized input (token IDs)
2. Per-layer hidden states
3. Final logits
4. Generated output text

Usage:
    python3 scripts/generate_golden_lfm2.py [--model-path ./models/lfm2-1.2b-thinking]

The output is saved to tests/golden/lfm2/ as binary files.
"""

import argparse
import json
import os
import struct
import sys
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Generate golden data for LFM2")
    parser.add_argument("--model-path", default="./models/lfm2-1.2b-thinking",
                       help="Path to model directory")
    parser.add_argument("--output-dir", default="./tests/golden/lfm2",
                       help="Output directory for golden data")
    parser.add_argument("--prompt", default="What is 2+2?",
                       help="Test prompt")
    parser.add_argument("--max-tokens", type=int, default=32,
                       help="Max tokens to generate")
    args = parser.parse_args()

    # Lazy imports for environments where these may not be installed
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: torch and transformers are required.")
        print("Install: pip install torch transformers")
        sys.exit(1)

    print(f"Model path: {args.model_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Prompt: {args.prompt}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    print(f"  BOS token: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")

    # Format with ChatML
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": args.prompt},
    ]

    # Apply chat template
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"\nChat template applied:")
    print(f"  {repr(chat_text[:200])}")

    # Tokenize
    input_ids = tokenizer.encode(chat_text, return_tensors="pt")
    print(f"\nTokenized: {input_ids.shape[1]} tokens")
    print(f"  First 20 tokens: {input_ids[0][:20].tolist()}")

    # Save token IDs
    token_ids = input_ids[0].tolist()
    with open(os.path.join(args.output_dir, "input_tokens.json"), "w") as f:
        json.dump({
            "prompt": args.prompt,
            "chat_text": chat_text,
            "token_ids": token_ids,
            "num_tokens": len(token_ids),
        }, f, indent=2)

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model.eval()
    print(f"  Model type: {model.config.model_type}")
    print(f"  Layers: {model.config.num_hidden_layers}")
    print(f"  Hidden size: {model.config.hidden_size}")
    print(f"  Device: {next(model.parameters()).device}")

    # Forward pass to get hidden states
    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(input_ids.to(model.device), output_hidden_states=True)

    # Save hidden states (per-layer, last token only to save space)
    hidden_states = outputs.hidden_states  # tuple of (num_layers+1) tensors
    print(f"  Got {len(hidden_states)} hidden states (embedding + {len(hidden_states)-1} layers)")

    for i, hs in enumerate(hidden_states):
        # Save last token hidden state as FP32
        last_token_hs = hs[0, -1, :].float().cpu().numpy()
        save_path = os.path.join(args.output_dir, f"hidden_state_layer_{i}.bin")
        last_token_hs.tofile(save_path)
        print(f"  Layer {i}: shape={hs.shape}, saved last token ({last_token_hs.shape}), "
              f"norm={np.linalg.norm(last_token_hs):.4f}")

    # Save final logits (last token)
    logits = outputs.logits[0, -1, :].float().cpu().numpy()
    logits_path = os.path.join(args.output_dir, "logits.bin")
    logits.tofile(logits_path)
    print(f"\n  Logits shape: {outputs.logits.shape}")
    print(f"  Top-5 tokens: {np.argsort(logits)[-5:][::-1].tolist()}")
    top_token = np.argmax(logits)
    print(f"  Top token: {top_token} = '{tokenizer.decode([top_token])}'")

    # Generate text
    print(f"\nGenerating {args.max_tokens} tokens...")
    with torch.no_grad():
        generated = model.generate(
            input_ids.to(model.device),
            max_new_tokens=args.max_tokens,
            temperature=0.01,  # Near-greedy for determinism
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=False)
    print(f"\n  Generated text: {repr(generated_text[:200])}")

    # Save generation result
    with open(os.path.join(args.output_dir, "generation.json"), "w") as f:
        json.dump({
            "prompt": args.prompt,
            "generated_tokens": generated[0][input_ids.shape[1]:].tolist(),
            "generated_text": generated_text,
            "num_generated": len(generated[0]) - input_ids.shape[1],
            "eos_token_id": tokenizer.eos_token_id,
        }, f, indent=2)

    # Save model config summary
    config_dict = {}
    for key in ["model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "vocab_size",
                "layer_types", "conv_L_cache", "eos_token_id", "bos_token_id", "dtype"]:
        try:
            val = getattr(model.config, key, None)
            if val is not None:
                config_dict[key] = val
        except Exception:
            pass
    with open(os.path.join(args.output_dir, "model_info.json"), "w") as f:
        json.dump(config_dict, f, indent=2, default=str)

    print(f"\nGolden data saved to {args.output_dir}/")
    print("Files:")
    for f in sorted(os.listdir(args.output_dir)):
        size = os.path.getsize(os.path.join(args.output_dir, f))
        print(f"  {f}: {size:,} bytes")

    print("\nDone!")


if __name__ == "__main__":
    main()
