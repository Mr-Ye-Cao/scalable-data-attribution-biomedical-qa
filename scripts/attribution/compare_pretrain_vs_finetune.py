#!/usr/bin/env python3
"""
Compare influence from PRETRAIN vs FINETUNE data on test queries.

Uses:
- Test gradients: NEW aligned (actual model outputs)
- Finetune gradients: NEW aligned (proper instruction/output split)
- Pretrain gradients: From rapidin_original (raw text - 18,353 samples)

This answers: "Is the model's prediction driven more by pretrain or finetune data?"
"""

import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path("/lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune")

# Paths
ALIGNED_GRADS = BASE_DIR / "results/rapidin_aligned/grads"
ORIGINAL_GRADS = BASE_DIR / "results/rapidin_original/grads"

PATHS = {
    "baseline": {
        "test": ALIGNED_GRADS / "baseline/test",
        "finetune": ALIGNED_GRADS / "baseline/finetune",
        "pretrain": ORIGINAL_GRADS / "baseline/pretrain",
    },
    "ckpt32": {
        "test": ALIGNED_GRADS / "ckpt32/test",
        "finetune": ALIGNED_GRADS / "ckpt32/finetune",
        "pretrain": ORIGINAL_GRADS / "ckpt32/pretrain",
    },
    "ckpt64": {
        "test": ALIGNED_GRADS / "ckpt64/test",
        "finetune": ALIGNED_GRADS / "ckpt64/finetune",
        "pretrain": ORIGINAL_GRADS / "ckpt64/pretrain",
    },
    "ckpt96": {
        "test": ALIGNED_GRADS / "ckpt96/test",
        "finetune": ALIGNED_GRADS / "ckpt96/finetune",
        "pretrain": ORIGINAL_GRADS / "ckpt96/pretrain",
    }
}


def load_gradient(path):
    """Load a single gradient file."""
    data = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(data, dict):
        return data.get('grad', data.get('gradient', list(data.values())[0]))
    return data


def load_gradients(grad_dir, max_samples=None):
    """Load gradients from a directory."""
    grad_dir = Path(grad_dir)
    if not grad_dir.exists():
        print(f"WARNING: {grad_dir} not found")
        return None

    files = sorted(grad_dir.glob("*.pt"))
    if max_samples:
        files = files[:max_samples]

    if len(files) == 0:
        print(f"WARNING: No .pt files in {grad_dir}")
        return None

    gradients = []
    for f in tqdm(files, desc=f"Loading {grad_dir.name}"):
        grad = load_gradient(f)
        if isinstance(grad, torch.Tensor):
            gradients.append(grad.float())
        else:
            gradients.append(torch.tensor(grad).float())

    return torch.stack(gradients)


def compute_influence(test_grads, train_grads):
    """Compute normalized influence scores."""
    test_norm = test_grads / (test_grads.norm(dim=1, keepdim=True) + 1e-8)
    train_norm = train_grads / (train_grads.norm(dim=1, keepdim=True) + 1e-8)
    return test_norm @ train_norm.T


def main():
    print("=" * 70)
    print("Comparing PRETRAIN vs FINETUNE Influence")
    print("=" * 70)

    results = {}

    for model_name in ["baseline", "ckpt32", "ckpt64", "ckpt96"]:
        print(f"\n\n{'#'*70}")
        print(f"# {model_name.upper()}")
        print(f"{'#'*70}")

        paths = PATHS[model_name]

        # Load test gradients (aligned - 50 samples)
        print("\nLoading TEST gradients (aligned)...")
        test_grads = load_gradients(paths["test"])
        if test_grads is None:
            continue
        print(f"  Shape: {test_grads.shape}")

        # Load finetune gradients (aligned - 50 samples)
        print("\nLoading FINETUNE gradients (aligned)...")
        finetune_grads = load_gradients(paths["finetune"])
        if finetune_grads is None:
            continue
        print(f"  Shape: {finetune_grads.shape}")

        # Load pretrain gradients (original - use first 50 for fair comparison)
        print("\nLoading PRETRAIN gradients (first 50)...")
        pretrain_grads = load_gradients(paths["pretrain"], max_samples=50)
        if pretrain_grads is None:
            continue
        print(f"  Shape: {pretrain_grads.shape}")

        # Compute influence scores
        print("\nComputing influence scores...")
        finetune_infl = compute_influence(test_grads, finetune_grads)
        pretrain_infl = compute_influence(test_grads, pretrain_grads)

        # Statistics
        ft_stats = {
            "mean": float(finetune_infl.mean()),
            "std": float(finetune_infl.std()),
            "max": float(finetune_infl.max()),
            "max_per_query": float(finetune_infl.max(dim=1).values.mean()),
        }
        pt_stats = {
            "mean": float(pretrain_infl.mean()),
            "std": float(pretrain_infl.std()),
            "max": float(pretrain_infl.max()),
            "max_per_query": float(pretrain_infl.max(dim=1).values.mean()),
        }

        print(f"\n{'='*60}")
        print(f"{model_name.upper()} RESULTS")
        print(f"{'='*60}")
        print(f"\nFinetune Influence (50 test x 50 finetune):")
        print(f"  Mean: {ft_stats['mean']:.6f}")
        print(f"  Max:  {ft_stats['max']:.6f}")
        print(f"  Max per query: {ft_stats['max_per_query']:.6f}")

        print(f"\nPretrain Influence (50 test x 50 pretrain):")
        print(f"  Mean: {pt_stats['mean']:.6f}")
        print(f"  Max:  {pt_stats['max']:.6f}")
        print(f"  Max per query: {pt_stats['max_per_query']:.6f}")

        # Compare: which has higher influence per query?
        ft_max = finetune_infl.max(dim=1).values
        pt_max = pretrain_infl.max(dim=1).values
        ft_wins = (ft_max > pt_max).sum().item()
        pt_wins = (pt_max > ft_max).sum().item()

        print(f"\nPer-query comparison (max influence):")
        print(f"  Finetune wins: {ft_wins}/{len(test_grads)} ({100*ft_wins/len(test_grads):.1f}%)")
        print(f"  Pretrain wins: {pt_wins}/{len(test_grads)} ({100*pt_wins/len(test_grads):.1f}%)")

        results[model_name] = {
            "finetune": ft_stats,
            "pretrain": pt_stats,
            "finetune_wins": ft_wins,
            "pretrain_wins": pt_wins,
            "num_test": len(test_grads),
        }

    # Summary comparison
    if len(results) >= 2:
        print(f"\n\n{'#'*70}")
        print("# SUMMARY: PRETRAIN vs FINETUNE ACROSS TRAINING")
        print(f"{'#'*70}")

        accuracy = {"baseline": "41.8%", "ckpt32": "69.0%", "ckpt64": "74.4%", "ckpt96": "69.0%"}

        print("\n## Influence Comparison Table:")
        print(f"  {'Model':<10} {'Accuracy':<10} {'Finetune':<12} {'Pretrain':<12} {'FT/PT Ratio':<12} {'FT Wins':<10}")
        print(f"  {'-'*10} {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

        for model_name in ["baseline", "ckpt32", "ckpt64", "ckpt96"]:
            if model_name not in results:
                continue
            r = results[model_name]
            ft_mean = r['finetune']['mean']
            pt_mean = r['pretrain']['mean']
            ratio = ft_mean / pt_mean if pt_mean > 0 else float('inf')
            ft_wins = r['finetune_wins']
            num_test = r['num_test']
            print(f"  {model_name:<10} {accuracy[model_name]:<10} {ft_mean:<12.4f} {pt_mean:<12.4f} {ratio:<12.1f}x {ft_wins}/{num_test}")

        # Changes from baseline
        if "baseline" in results:
            baseline_ft = results['baseline']['finetune']['mean']
            baseline_pt = results['baseline']['pretrain']['mean']

            print("\n## Changes from Baseline:")
            print(f"  {'Model':<10} {'FT Change':<15} {'PT Change':<15}")
            print(f"  {'-'*10} {'-'*15} {'-'*15}")

            for model_name in ["baseline", "ckpt32", "ckpt64", "ckpt96"]:
                if model_name not in results:
                    continue
                r = results[model_name]
                ft_change = (r['finetune']['mean'] - baseline_ft) / baseline_ft * 100
                pt_change = (r['pretrain']['mean'] - baseline_pt) / baseline_pt * 100
                if model_name == "baseline":
                    print(f"  {model_name:<10} {'-':<15} {'-':<15}")
                else:
                    print(f"  {model_name:<10} {ft_change:+.1f}%{'':<10} {pt_change:+.1f}%")

    # Save results
    output_path = BASE_DIR / "results/rapidin_aligned/pretrain_vs_finetune_comparison.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
