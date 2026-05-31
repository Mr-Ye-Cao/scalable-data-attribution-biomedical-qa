#!/usr/bin/env python3
"""
GOAL3 Experiment 1: TracIn Memory Estimation for OLMo-3-7B

Compute gradients for 3-5 samples and estimate total storage requirements
to demonstrate that full TracIn is infeasible at LLM scale.

Theoretical estimate:
- OLMo-3-7B: ~7B parameters
- Gradient size per sample (FP16): 7B × 2 bytes = 14 GB
- Total for 2000 samples: 28 TB (infeasible)

This script validates this estimate with actual measurements.

Usage:
    python scripts/attribution/goal3/tracin_memory_estimation.py --n-samples 5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

BASE_DIR = Path("/lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune")
MODEL_PATH = BASE_DIR / "OLMo-3-7B-Instruct"
FINETUNE_DATA = BASE_DIR / "data-aggregate/input/finetune/finetune_500.jsonl"
OUTPUT_DIR = BASE_DIR / "results/goal3/tracin_memory"


def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_gradient_for_sample(model, tokenizer, sample, device):
    """Compute gradient for a single sample and return gradient tensors."""
    model.zero_grad()

    # Tokenize
    text = sample['instruction'] + sample['output']
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass
    outputs = model(**inputs, labels=inputs['input_ids'])
    loss = outputs.loss

    # Backward pass
    loss.backward()

    # Collect gradients
    gradients = {}
    total_elements = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.clone()
            total_elements += param.grad.numel()

    return gradients, total_elements, loss.item()


def estimate_gradient_size(gradients, dtype=torch.float16):
    """Estimate storage size for gradients in bytes."""
    bytes_per_element = 2 if dtype == torch.float16 else 4
    total_bytes = sum(g.numel() * bytes_per_element for g in gradients.values())
    return total_bytes


def save_gradients_to_disk(gradients, output_path, dtype=torch.float16):
    """Save gradients to disk and measure actual file size."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert to specified dtype and save
    gradients_to_save = {k: v.to(dtype) for k, v in gradients.items()}
    torch.save(gradients_to_save, output_path)

    # Get actual file size
    file_size = os.path.getsize(output_path)
    return file_size


def load_samples(n_samples):
    """Load n samples from finetune data."""
    samples = []
    with open(FINETUNE_DATA) as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            samples.append(json.loads(line))
    return samples


def format_bytes(size_bytes):
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def main():
    parser = argparse.ArgumentParser(description="TracIn memory estimation for OLMo-3-7B")
    parser.add_argument('--n-samples', type=int, default=5, help='Number of samples to compute gradients for')
    parser.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'fp32'], help='Gradient storage dtype')
    parser.add_argument('--save-gradients', action='store_true', help='Save gradients to disk to measure actual file size')
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == 'fp16' else torch.float32

    print("=" * 80)
    print("GOAL3 Experiment 1: TracIn Memory Estimation")
    print("=" * 80)

    # Load model
    print(f"\nLoading model from {MODEL_PATH}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Count parameters
    n_params = count_parameters(model)
    print(f"\nModel parameters: {n_params:,} ({n_params/1e9:.2f}B)")

    # Theoretical gradient size
    bytes_per_param = 2 if args.dtype == 'fp16' else 4
    theoretical_size = n_params * bytes_per_param
    print(f"Theoretical gradient size per sample ({args.dtype}): {format_bytes(theoretical_size)}")

    # Load samples
    samples = load_samples(args.n_samples)
    print(f"\nLoaded {len(samples)} samples for gradient computation")

    # Compute gradients for each sample
    print(f"\n{'=' * 80}")
    print(f"Computing gradients for {len(samples)} samples...")
    print(f"{'=' * 80}")

    results = []
    gradient_sizes = []
    file_sizes = []

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(tqdm(samples, desc="Computing gradients")):
        start_time = time.time()

        gradients, total_elements, loss = compute_gradient_for_sample(model, tokenizer, sample, device)

        compute_time = time.time() - start_time

        # Estimate memory size
        memory_size = estimate_gradient_size(gradients, dtype)
        gradient_sizes.append(memory_size)

        # Optionally save to disk
        if args.save_gradients:
            output_path = OUTPUT_DIR / f"gradient_sample_{i}.pt"
            file_size = save_gradients_to_disk(gradients, output_path, dtype)
            file_sizes.append(file_size)
            print(f"  Sample {i}: loss={loss:.4f}, memory={format_bytes(memory_size)}, "
                  f"file={format_bytes(file_size)}, time={compute_time:.2f}s")
        else:
            print(f"  Sample {i}: loss={loss:.4f}, memory={format_bytes(memory_size)}, time={compute_time:.2f}s")

        results.append({
            'sample_idx': i,
            'loss': loss,
            'memory_size_bytes': memory_size,
            'compute_time_sec': compute_time,
            'total_elements': total_elements,
        })

        # Clear gradients
        del gradients
        torch.cuda.empty_cache()

    # Summary statistics
    avg_memory = sum(gradient_sizes) / len(gradient_sizes)

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"\nModel: OLMo-3-7B ({n_params/1e9:.2f}B parameters)")
    print(f"Gradient dtype: {args.dtype}")
    print(f"Samples computed: {len(samples)}")

    print(f"\n--- Per-Sample Gradient Size ---")
    print(f"Theoretical (based on parameter count): {format_bytes(theoretical_size)}")
    print(f"Actual average: {format_bytes(avg_memory)}")

    if file_sizes:
        avg_file = sum(file_sizes) / len(file_sizes)
        print(f"Actual file size (saved to disk): {format_bytes(avg_file)}")

    # Extrapolate to full dataset
    print(f"\n--- Extrapolated Storage Requirements ---")

    datasets = {
        'Finetune (500 samples)': 500,
        'Pretrain (500 samples)': 500,
        'Total training (1000 samples)': 1000,
        'Full scale (10000 samples)': 10000,
    }

    print(f"\n| Dataset | Samples | Storage Required |")
    print(f"|---------|---------|------------------|")
    for name, n in datasets.items():
        total_storage = avg_memory * n
        print(f"| {name} | {n:,} | {format_bytes(total_storage)} |")

    # Comparison with RapidIn
    print(f"\n--- Comparison with RapidIn ---")
    rapidin_k = 2**16  # Typical RapidIn compression
    rapidin_size = n_params * 4 / rapidin_k  # FP32 / K
    print(f"RapidIn (K={rapidin_k}): {format_bytes(rapidin_size * 500)} for 500 samples")
    print(f"TracIn: {format_bytes(avg_memory * 500)} for 500 samples")
    print(f"Compression ratio: {avg_memory / rapidin_size:.0f}x")

    # Save results
    summary = {
        'model': 'OLMo-3-7B-Instruct',
        'n_params': n_params,
        'dtype': args.dtype,
        'n_samples_measured': len(samples),
        'theoretical_size_bytes': theoretical_size,
        'actual_avg_size_bytes': avg_memory,
        'per_sample_results': results,
        'storage_estimates': {
            'finetune_500': avg_memory * 500,
            'pretrain_500': avg_memory * 500,
            'total_1000': avg_memory * 1000,
            'total_10000': avg_memory * 10000,
        },
        'rapidin_comparison': {
            'rapidin_k': rapidin_k,
            'rapidin_size_500_samples': rapidin_size * 500,
            'tracin_size_500_samples': avg_memory * 500,
            'compression_ratio': avg_memory / rapidin_size,
        }
    }

    summary_file = OUTPUT_DIR / "tracin_memory_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")

    print(f"\n{'=' * 80}")
    print("CONCLUSION")
    print(f"{'=' * 80}")
    print(f"""
TracIn requires storing full gradients for each training sample:
- Per sample: {format_bytes(avg_memory)}
- For 1000 training samples: {format_bytes(avg_memory * 1000)}
- For 10000 samples: {format_bytes(avg_memory * 10000)}

This is INFEASIBLE for large-scale attribution studies.

RapidIn compresses gradients by {avg_memory / rapidin_size:.0f}x, making it practical:
- Per sample: {format_bytes(rapidin_size)}
- For 1000 samples: {format_bytes(rapidin_size * 1000)}
""")


if __name__ == "__main__":
    main()
