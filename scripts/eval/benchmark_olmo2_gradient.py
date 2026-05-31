#!/usr/bin/env python3
"""
Benchmark gradient computation speed for OLMo-2-1B.

This script measures how fast we can compute gradients for training data attribution,
comparing to OLMo-3-7B (~28.6 sec/sample with CPU offload on RTX 5090).

Expected speedup: ~20x (24s -> ~1s per sample)
"""

import os
import sys
import time
import json
import torch
import gc
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add RapidIn to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'RapidIn'))
from RapidIn.RapidGrad import RapidGrad


def load_model_and_tokenizer(model_path, device="cuda:0"):
    """Load OLMo-2-1B model and tokenizer."""
    print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    model.train()  # Enable gradient computation

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model loaded: {total_params/1e9:.2f}B total params, {trainable_params/1e9:.2f}B trainable")

    return model, tokenizer


def get_trainable_params(model):
    """Get list of trainable parameters with dim >= 2."""
    params = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.dim() >= 2:
            params.append(p)
    return params


def compute_gradient(model, input_ids, labels, device="cuda:0"):
    """Compute gradient for a single sample."""
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    # Forward pass
    outputs = model(input_ids)
    logits = outputs.logits

    # Compute loss (cross-entropy)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # Backward pass
    params = get_trainable_params(model)
    grads = torch.autograd.grad(loss, params)

    # Flatten and concatenate gradients
    grad_vec = torch.cat([g.reshape(-1) for g in grads])

    model.zero_grad(set_to_none=True)

    return grad_vec, loss.item()


def load_sample_data(data_path, tokenizer, n_samples=5, max_length=256):
    """Load a few samples from the health data."""
    samples = []

    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            data = json.loads(line)
            text = data.get('text', '')[:2000]  # Truncate long texts

            # Tokenize
            encoded = tokenizer(
                text,
                return_tensors='pt',
                max_length=max_length,
                truncation=True,
                padding='max_length'
            )

            samples.append({
                'input_ids': encoded['input_ids'],
                'labels': encoded['input_ids'].clone(),  # For LM, labels = input_ids
                'text_preview': text[:100]
            })

    return samples


def benchmark_gradient_computation(model, tokenizer, samples, device="cuda:0", warmup=2, n_runs=10):
    """Benchmark gradient computation speed."""
    print(f"\n{'='*60}")
    print("Gradient Computation Benchmark")
    print(f"{'='*60}")

    # Get gradient dimension
    params = get_trainable_params(model)
    total_grad_dim = sum(p.numel() for p in params)
    print(f"Gradient dimension: {total_grad_dim:,} ({total_grad_dim/1e9:.3f}B)")

    # Warmup
    print(f"\nWarmup ({warmup} iterations)...")
    for i in range(warmup):
        sample = samples[i % len(samples)]
        grad_vec, loss = compute_gradient(model, sample['input_ids'], sample['labels'], device)
        del grad_vec
        torch.cuda.empty_cache()

    # Benchmark
    print(f"\nBenchmarking ({n_runs} iterations)...")
    times = []
    for i in range(n_runs):
        sample = samples[i % len(samples)]

        torch.cuda.synchronize()
        start = time.perf_counter()

        grad_vec, loss = compute_gradient(model, sample['input_ids'], sample['labels'], device)

        torch.cuda.synchronize()
        end = time.perf_counter()

        elapsed = end - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.4f}s, loss={loss:.4f}, grad_shape={grad_vec.shape}")

        del grad_vec
        torch.cuda.empty_cache()

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"\n{'='*60}")
    print("Results:")
    print(f"  Average time: {avg_time:.4f}s")
    print(f"  Min time: {min_time:.4f}s")
    print(f"  Max time: {max_time:.4f}s")
    print(f"  Throughput: {1/avg_time:.2f} samples/sec")
    print(f"\nComparison to OLMo-3-7B (28.6s/sample with CPU offload):")
    print(f"  Speedup: {28.6/avg_time:.1f}x")
    print(f"{'='*60}")

    return times


def benchmark_with_rapidgrad(model, tokenizer, samples, device="cuda:0", rapidgrad_k=65536):
    """Benchmark with RapidGrad compression."""
    print(f"\n{'='*60}")
    print(f"RapidGrad Compression Benchmark (K={rapidgrad_k})")
    print(f"{'='*60}")

    # Create mock config for RapidGrad
    class MockConfig:
        class influence:
            class RapidGrad:
                enable = True
                RapidGrad_K = rapidgrad_k
                shuffle_lambda = 20

    rapidgrad = RapidGrad(MockConfig(), device)

    # Benchmark with compression
    times = []
    for i in range(5):
        sample = samples[i % len(samples)]

        torch.cuda.synchronize()
        start = time.perf_counter()

        # Compute gradient
        grad_vec, loss = compute_gradient(model, sample['input_ids'], sample['labels'], device)

        # Compress with RapidGrad
        compressed_grad = rapidgrad(grad_vec, rapidgrad_k)

        torch.cuda.synchronize()
        end = time.perf_counter()

        elapsed = end - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.4f}s, compressed_shape={compressed_grad.shape}")

        del grad_vec, compressed_grad
        torch.cuda.empty_cache()

    avg_time = sum(times) / len(times)
    print(f"\nAverage time with RapidGrad: {avg_time:.4f}s")
    print(f"Speedup vs OLMo-3-7B: {28.6/avg_time:.1f}x")

    return times


def main():
    # Configuration
    model_path = "./OLMo-2-0425-1B"
    data_path = "./data/olmo_mix_1124/pes2o-0000-health.jsonl"
    device = "cuda:0"

    # Check GPU
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_path, device)

    # Check GPU memory after loading
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"GPU Memory after model load: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    # Load sample data
    print(f"\nLoading sample data from {data_path}...")
    samples = load_sample_data(data_path, tokenizer, n_samples=10, max_length=256)
    print(f"Loaded {len(samples)} samples")

    # Benchmark gradient computation
    times = benchmark_gradient_computation(model, tokenizer, samples, device)

    # Benchmark with RapidGrad
    try:
        benchmark_with_rapidgrad(model, tokenizer, samples, device)
    except Exception as e:
        print(f"RapidGrad benchmark failed: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
