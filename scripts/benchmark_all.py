#!/usr/bin/env python3
"""Benchmark NeuroGrid vs HuggingFace vs vLLM on LFM2.5-1.2B-Thinking."""
import time
import json
import subprocess
import sys

MODEL_PATH = "./models/lfm2-1.2b-thinking"
PROMPT = "Explain quantum computing in detail."
MAX_TOKENS = 128
N_RUNS = 5
WARMUP = 2

def benchmark_huggingface():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading HuggingFace model...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)
    model.eval()

    messages = [{"role": "user", "content": PROMPT}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tok.encode(text, return_tensors="pt").to("cuda:0")
    n_input = input_ids.shape[1]

    # Warmup
    for _ in range(WARMUP):
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=16, do_sample=False)

    results = []
    for i in range(N_RUNS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=MAX_TOKENS, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        n_gen = out.shape[1] - n_input
        tps = n_gen / elapsed
        results.append({"elapsed_ms": elapsed * 1000, "tokens": n_gen, "tps": tps})

    del model
    torch.cuda.empty_cache()
    return results

def benchmark_vllm():
    from vllm import LLM, SamplingParams

    print("Loading vLLM model...")
    try:
        llm = LLM(model=MODEL_PATH, trust_remote_code=True, dtype="bfloat16",
                   max_model_len=2048, gpu_memory_utilization=0.9)
    except Exception as e:
        print(f"  vLLM failed to load: {e}")
        return None

    params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0)

    messages = [{"role": "user", "content": PROMPT}]

    # Warmup
    for _ in range(WARMUP):
        llm.chat(messages, params)

    results = []
    for i in range(N_RUNS):
        start = time.perf_counter()
        out = llm.chat(messages, params)
        elapsed = time.perf_counter() - start
        n_gen = len(out[0].outputs[0].token_ids)
        tps = n_gen / elapsed
        results.append({"elapsed_ms": elapsed * 1000, "tokens": n_gen, "tps": tps})

    del llm
    import torch
    torch.cuda.empty_cache()
    return results

def benchmark_neurogrid():
    import requests
    print("Testing NeuroGrid (must be running on :8091)...")

    url = "http://localhost:8091/v1/chat/completions"
    payload = {
        "model": "lfm2-1.2b-thinking",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.01
    }

    # Warmup
    for _ in range(WARMUP):
        try:
            requests.post(url, json=payload, timeout=30)
        except:
            print("  NeuroGrid not running on :8091")
            return None

    results = []
    for i in range(N_RUNS):
        start = time.perf_counter()
        resp = requests.post(url, json=payload, timeout=60)
        elapsed = time.perf_counter() - start
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        n_gen = MAX_TOKENS  # approximate
        tps = n_gen / elapsed
        results.append({"elapsed_ms": elapsed * 1000, "tokens": n_gen, "tps": tps})

    return results

def print_results(name, results):
    if results is None:
        print(f"  {name:20s}: SKIPPED (not available)")
        return
    tps_list = [r["tps"] for r in results]
    ms_list = [r["elapsed_ms"] for r in results]
    avg_tps = sum(tps_list) / len(tps_list)
    avg_ms = sum(ms_list) / len(ms_list)
    min_ms = min(ms_list)
    print(f"  {name:20s}: {avg_tps:7.1f} tok/s | avg {avg_ms:7.0f}ms | best {min_ms:7.0f}ms | {MAX_TOKENS} tokens")

if __name__ == "__main__":
    print(f"Benchmark: LFM2.5-1.2B-Thinking ({MAX_TOKENS} tokens, {N_RUNS} runs)")
    print(f"Prompt: {PROMPT}")
    print()

    all_results = {}

    # 1. HuggingFace
    print("=" * 60)
    hf = benchmark_huggingface()
    all_results["huggingface"] = hf

    # 2. vLLM
    print("=" * 60)
    vllm_results = benchmark_vllm()
    all_results["vllm"] = vllm_results

    # 3. NeuroGrid
    print("=" * 60)
    ng = benchmark_neurogrid()
    all_results["neurogrid"] = ng

    # Results
    print()
    print("=" * 60)
    print(f"RESULTS ({MAX_TOKENS} tokens, {N_RUNS} runs, best of {N_RUNS})")
    print("=" * 60)
    print_results("HuggingFace", hf)
    print_results("vLLM", vllm_results)
    print_results("NeuroGrid", ng)

    # Save
    with open("benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to benchmark_results.json")
