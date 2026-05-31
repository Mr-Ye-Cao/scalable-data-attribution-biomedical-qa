#!/usr/bin/env python3
"""
Compare influence scores across baseline, checkpoint-32, and checkpoint-64 using ALIGNED data.

For each test query, computes:
1. Influence from finetune data (500 train samples)

Then compares how influence distributions change during fine-tuning.

NOTE: For pretrain influence, we need to prepare aligned pretrain data separately
since pretrain data is raw text without QA format.
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Paths
BASE_DIR = Path("/lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune")
GRADS_DIR = BASE_DIR / "results/rapidin_aligned/grads"

# Gradient paths for ALIGNED data
PATHS = {
    "baseline": {
        "test": GRADS_DIR / "baseline/test",
        "finetune": GRADS_DIR / "baseline/finetune",
    },
    "ckpt32": {
        "test": GRADS_DIR / "ckpt32/test",
        "finetune": GRADS_DIR / "ckpt32/finetune",
    },
    "ckpt64": {
        "test": GRADS_DIR / "ckpt64/test",
        "finetune": GRADS_DIR / "ckpt64/finetune",
    },
    "ckpt96": {
        "test": GRADS_DIR / "ckpt96/test",
        "finetune": GRADS_DIR / "ckpt96/finetune",
    }
}


def load_gradient(path):
    """Load a single gradient file."""
    data = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(data, dict):
        return data.get('grad', data.get('gradient', list(data.values())[0]))
    return data


def load_all_gradients(grad_dir, max_samples=None):
    """Load all gradients from a directory."""
    grad_dir = Path(grad_dir)
    if not grad_dir.exists():
        print(f"WARNING: Directory not found: {grad_dir}")
        return None

    files = sorted(grad_dir.glob("*.pt"))
    if len(files) == 0:
        print(f"WARNING: No .pt files found in {grad_dir}")
        return None

    if max_samples:
        files = files[:max_samples]

    gradients = []
    for f in tqdm(files, desc=f"Loading {grad_dir.name}"):
        grad = load_gradient(f)
        if isinstance(grad, torch.Tensor):
            gradients.append(grad.float())
        else:
            gradients.append(torch.tensor(grad).float())

    return torch.stack(gradients)  # [N, K]


def compute_influence_matrix(test_grads, train_grads):
    """Compute influence scores: test_grads @ train_grads.T"""
    # Normalize for stability
    test_norm = test_grads / (test_grads.norm(dim=1, keepdim=True) + 1e-8)
    train_norm = train_grads / (train_grads.norm(dim=1, keepdim=True) + 1e-8)

    # Compute dot products
    scores = test_norm @ train_norm.T  # [num_test, num_train]
    return scores


def analyze_influence(scores, name):
    """Analyze influence score distribution."""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")

    # Overall stats
    print(f"Shape: {scores.shape}")
    print(f"Mean: {scores.mean():.6f}")
    print(f"Std: {scores.std():.6f}")
    print(f"Min: {scores.min():.6f}")
    print(f"Max: {scores.max():.6f}")

    # Per-test-query stats (max influence per query)
    max_per_query = scores.max(dim=1).values
    print(f"\nPer-query max influence:")
    print(f"  Mean: {max_per_query.mean():.6f}")
    print(f"  Std: {max_per_query.std():.6f}")

    # Top-k analysis
    topk_mean = scores.topk(10, dim=1).values.mean(dim=1)
    print(f"\nTop-10 mean influence per query:")
    print(f"  Mean: {topk_mean.mean():.6f}")
    print(f"  Std: {topk_mean.std():.6f}")

    return {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "max_per_query_mean": float(max_per_query.mean()),
        "max_per_query_std": float(max_per_query.std()),
        "top10_mean": float(topk_mean.mean()),
    }


def main():
    print("=" * 70)
    print("Comparing Influence Scores Across Training (ALIGNED DATA)")
    print("Baseline -> Ckpt32 (Epoch 1) -> Ckpt64 (Epoch 2) -> Ckpt96 (Epoch 3)")
    print("=" * 70)

    results = {}

    for model_name in ["baseline", "ckpt32", "ckpt64", "ckpt96"]:
        print(f"\n\n{'#'*70}")
        print(f"# Loading {model_name.upper()} gradients")
        print(f"{'#'*70}")

        paths = PATHS[model_name]

        # Load gradients
        print("\nLoading test gradients (500)...")
        test_grads = load_all_gradients(paths["test"])
        if test_grads is None:
            print(f"Skipping {model_name} - test gradients not found")
            continue
        print(f"Test gradients shape: {test_grads.shape}")

        print("\nLoading finetune gradients (500)...")
        finetune_grads = load_all_gradients(paths["finetune"])
        if finetune_grads is None:
            print(f"Skipping {model_name} - finetune gradients not found")
            continue
        print(f"Finetune gradients shape: {finetune_grads.shape}")

        # Compute influence matrices
        print("\nComputing influence scores...")

        finetune_influence = compute_influence_matrix(test_grads, finetune_grads)

        # Analyze
        results[model_name] = {
            "finetune": analyze_influence(
                finetune_influence,
                f"{model_name.upper()} - Finetune Influence (500 test x 500 train)"
            ),
        }

        # Self-influence analysis (test query influenced by itself in training)
        # This shouldn't happen since test/train are disjoint, but let's check diagonal
        print(f"\n{'-'*60}")
        print(f"Influence Statistics for {model_name.upper()}")
        print(f"{'-'*60}")

        # Find which finetune samples have highest influence on each test query
        top_indices = finetune_influence.topk(5, dim=1).indices
        print(f"\nTop-5 influential finetune indices for first 5 test queries:")
        for i in range(min(5, len(test_grads))):
            print(f"  Test {i}: {top_indices[i].tolist()}")

    if len(results) < 2:
        print("\nCannot compare - not enough gradients available")
        print("Run gradient computation first")
        return

    # Cross-model comparison
    print(f"\n\n{'#'*70}")
    print("# INFLUENCE COMPARISON ACROSS TRAINING (ALIGNED DATA)")
    print(f"{'#'*70}")

    print("\n## Finetune Influence Changes:")
    print(f"  {'Model':<12} {'Mean':>10} {'Change':>12} {'Max/Query':>12} {'Top-10':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*12} {'-'*10}")

    baseline_mean = results.get('baseline', {}).get('finetune', {}).get('mean', 0)
    for model_name in ["baseline", "ckpt32", "ckpt64", "ckpt96"]:
        if model_name not in results:
            continue
        stats = results[model_name]['finetune']
        change = stats['mean'] - baseline_mean
        change_pct = (change / baseline_mean * 100) if baseline_mean != 0 else 0
        change_str = f"{change:+.4f} ({change_pct:+.1f}%)" if model_name != "baseline" else "-"
        print(f"  {model_name:<12} {stats['mean']:>10.4f} {change_str:>12} {stats['max_per_query_mean']:>12.4f} {stats['top10_mean']:>10.4f}")

    # Summary table
    print("\n## Summary Table:")
    print(f"  | Model | Accuracy | Mean Influence | Change |")
    print(f"  |-------|----------|----------------|--------|")
    accuracy = {"baseline": "41.8%", "ckpt32": "69.0%", "ckpt64": "74.4%", "ckpt96": "69.0%"}
    for model_name in ["baseline", "ckpt32", "ckpt64", "ckpt96"]:
        if model_name not in results:
            continue
        stats = results[model_name]['finetune']
        change = stats['mean'] - baseline_mean
        change_pct = (change / baseline_mean * 100) if baseline_mean != 0 else 0
        change_str = f"{change_pct:+.1f}%" if model_name != "baseline" else "-"
        print(f"  | {model_name} | {accuracy[model_name]} | {stats['mean']:.4f} | {change_str} |")

    # Save results
    output_path = BASE_DIR / "results/rapidin_aligned/aligned_influence_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
