#!/usr/bin/env python3
"""
Compare influence scores between baseline and checkpoint-64 models.

For each test query, computes:
1. Influence from pretrain data (18,353 candidates)
2. Influence from finetune data (500 train samples)

Compares how influence distributions change after fine-tuning.
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Paths
BASE_DIR = Path("/lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune")
GRADS_DIR = BASE_DIR / "results/rapidin_original/grads"

# Gradient paths
PATHS = {
    "baseline": {
        "test": GRADS_DIR / "baseline/test",
        "finetune": GRADS_DIR / "baseline/finetune",
        "pretrain": GRADS_DIR / "baseline/pretrain",
    },
    "ckpt64": {
        "test": GRADS_DIR / "ckpt64/test",
        "finetune": GRADS_DIR / "ckpt64/finetune",
        "pretrain": GRADS_DIR / "ckpt64/pretrain",
    }
}

def load_gradient(path):
    """Load a single gradient file."""
    data = torch.load(path, map_location='cpu')
    if isinstance(data, dict):
        return data.get('grad', data.get('gradient', list(data.values())[0]))
    return data

def load_all_gradients(grad_dir, max_samples=None):
    """Load all gradients from a directory."""
    grad_dir = Path(grad_dir)
    files = sorted(grad_dir.glob("*.pt"))
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
    print(f"Mean: {scores.mean():.4f}")
    print(f"Std: {scores.std():.4f}")
    print(f"Min: {scores.min():.4f}")
    print(f"Max: {scores.max():.4f}")

    # Per-test-query stats (max influence per query)
    max_per_query = scores.max(dim=1).values
    print(f"\nPer-query max influence:")
    print(f"  Mean: {max_per_query.mean():.4f}")
    print(f"  Std: {max_per_query.std():.4f}")

    # Top-k analysis
    topk_mean = scores.topk(10, dim=1).values.mean(dim=1)
    print(f"\nTop-10 mean influence per query:")
    print(f"  Mean: {topk_mean.mean():.4f}")
    print(f"  Std: {topk_mean.std():.4f}")

    return {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "max_per_query_mean": float(max_per_query.mean()),
        "top10_mean": float(topk_mean.mean()),
    }

def main():
    print("="*60)
    print("Comparing Baseline vs Checkpoint-64 Influence Scores")
    print("="*60)

    results = {}

    for model_name in ["baseline", "ckpt64"]:
        print(f"\n\n{'#'*60}")
        print(f"# Loading {model_name.upper()} gradients")
        print(f"{'#'*60}")

        paths = PATHS[model_name]

        # Load gradients
        print("\nLoading test gradients (500)...")
        test_grads = load_all_gradients(paths["test"])
        print(f"Test gradients shape: {test_grads.shape}")

        print("\nLoading finetune gradients (500)...")
        finetune_grads = load_all_gradients(paths["finetune"])
        print(f"Finetune gradients shape: {finetune_grads.shape}")

        print("\nLoading pretrain gradients (18,353)...")
        pretrain_grads = load_all_gradients(paths["pretrain"])
        print(f"Pretrain gradients shape: {pretrain_grads.shape}")

        # Compute influence matrices
        print("\nComputing influence scores...")

        finetune_influence = compute_influence_matrix(test_grads, finetune_grads)
        pretrain_influence = compute_influence_matrix(test_grads, pretrain_grads)

        # Analyze
        results[model_name] = {
            "finetune": analyze_influence(finetune_influence, f"{model_name.upper()} - Finetune Influence (500 test x 500 train)"),
            "pretrain": analyze_influence(pretrain_influence, f"{model_name.upper()} - Pretrain Influence (500 test x 18,353 pretrain)"),
        }

        # Compare finetune vs pretrain for this model
        print(f"\n{'-'*60}")
        print(f"{model_name.upper()}: Finetune vs Pretrain comparison")
        print(f"{'-'*60}")
        ft_max = finetune_influence.max(dim=1).values
        pt_max = pretrain_influence.max(dim=1).values

        ft_wins = (ft_max > pt_max).sum().item()
        pt_wins = (pt_max > ft_max).sum().item()

        print(f"Queries where finetune has higher max influence: {ft_wins}/500 ({100*ft_wins/500:.1f}%)")
        print(f"Queries where pretrain has higher max influence: {pt_wins}/500 ({100*pt_wins/500:.1f}%)")

        results[model_name]["finetune_wins"] = ft_wins
        results[model_name]["pretrain_wins"] = pt_wins

    # Cross-model comparison
    print(f"\n\n{'#'*60}")
    print("# BASELINE vs CKPT64 COMPARISON")
    print(f"{'#'*60}")

    print("\n## Finetune Influence Changes:")
    print(f"  Baseline mean: {results['baseline']['finetune']['mean']:.4f}")
    print(f"  Ckpt64 mean:   {results['ckpt64']['finetune']['mean']:.4f}")
    print(f"  Change:        {results['ckpt64']['finetune']['mean'] - results['baseline']['finetune']['mean']:+.4f}")

    print("\n## Pretrain Influence Changes:")
    print(f"  Baseline mean: {results['baseline']['pretrain']['mean']:.4f}")
    print(f"  Ckpt64 mean:   {results['ckpt64']['pretrain']['mean']:.4f}")
    print(f"  Change:        {results['ckpt64']['pretrain']['mean'] - results['baseline']['pretrain']['mean']:+.4f}")

    print("\n## Finetune vs Pretrain Dominance:")
    print(f"  Baseline: Finetune wins {results['baseline']['finetune_wins']}/500, Pretrain wins {results['baseline']['pretrain_wins']}/500")
    print(f"  Ckpt64:   Finetune wins {results['ckpt64']['finetune_wins']}/500, Pretrain wins {results['ckpt64']['pretrain_wins']}/500")

    # Save results
    output_path = BASE_DIR / "results/rapidin_original/baseline_vs_ckpt64_comparison.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
