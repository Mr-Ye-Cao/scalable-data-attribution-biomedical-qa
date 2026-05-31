#!/usr/bin/env python3
"""
Analyze Iteration 4: No Linebreaks in Context

Compare pretrain influence with line breaks removed from context.
Only baseline is computed for quick verification.
"""

import json
import torch
from pathlib import Path

BASE_DIR = Path("/lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune")

# Paths
ALIGNED_GRADS = BASE_DIR / "results/rapidin_aligned_500/grads"  # finetune & test
ITER3_GRADS = BASE_DIR / "results/controlled_experiment/iteration3/grads"  # full match (with linebreaks)
ITER4_GRADS = BASE_DIR / "results/controlled_experiment/iteration4/grads"  # no linebreaks

OUTPUT_DIR = BASE_DIR / "results/controlled_experiment/iteration4/analysis"


def load_gradient(path):
    data = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(data, dict):
        return data.get('grad', data.get('gradient', list(data.values())[0]))
    return data


def load_gradients(grad_dir, max_samples=500):
    grad_dir = Path(grad_dir)
    if not grad_dir.exists():
        return None
    files = sorted(grad_dir.glob("*.pt"))[:max_samples]
    if len(files) == 0:
        return None
    gradients = []
    for f in files:
        grad = load_gradient(f)
        if isinstance(grad, torch.Tensor):
            gradients.append(grad.float())
        else:
            gradients.append(torch.tensor(grad).float())
    return torch.stack(gradients)


def compute_influence(test_grads, train_grads):
    test_norm = test_grads / (test_grads.norm(dim=1, keepdim=True) + 1e-8)
    train_norm = train_grads / (train_grads.norm(dim=1, keepdim=True) + 1e-8)
    return test_norm @ train_norm.T


def main():
    print("=" * 80)
    print("ITERATION 4: NO LINEBREAKS vs WITH LINEBREAKS (Baseline Only)")
    print("=" * 80)

    model_name = "baseline"

    # Load gradients
    print(f"\nLoading gradients...")
    test_grads = load_gradients(ALIGNED_GRADS / model_name / "test")
    finetune_grads = load_gradients(ALIGNED_GRADS / model_name / "finetune")
    pt_iter3 = load_gradients(ITER3_GRADS / model_name / "pretrain")
    pt_iter4 = load_gradients(ITER4_GRADS / model_name / "pretrain")

    missing = []
    if test_grads is None: missing.append("test")
    if finetune_grads is None: missing.append("finetune")
    if pt_iter3 is None: missing.append("pretrain_iter3")
    if pt_iter4 is None: missing.append("pretrain_iter4")

    if missing:
        print(f"Missing gradients: {missing}")
        return

    print(f"  Test: {test_grads.shape}")
    print(f"  Finetune: {finetune_grads.shape}")
    print(f"  Pretrain (iter3/with linebreaks): {pt_iter3.shape}")
    print(f"  Pretrain (iter4/no linebreaks): {pt_iter4.shape}")

    # Compute influence
    print("\nComputing influence scores...")
    ft_infl = compute_influence(test_grads, finetune_grads)
    pt3_infl = compute_influence(test_grads, pt_iter3)
    pt4_infl = compute_influence(test_grads, pt_iter4)

    # Stats - max influence per test sample
    ft_max = ft_infl.max(dim=1).values
    pt3_max = pt3_infl.max(dim=1).values
    pt4_max = pt4_infl.max(dim=1).values

    ft_mean = float(ft_max.mean())
    pt3_mean = float(pt3_max.mean())
    pt4_mean = float(pt4_max.mean())

    # FT wins comparison
    ft_wins_iter3 = (ft_max > pt3_max).sum().item()
    ft_wins_iter4 = (ft_max > pt4_max).sum().item()

    # Compare
    print(f"\n{'='*70}")
    print(f"RESULTS: BASELINE")
    print(f"{'='*70}")
    print(f"{'Dataset':<40} {'Max Mean':<12} {'FT/PT Ratio':<12} {'FT Wins':<10}")
    print(f"{'-'*40} {'-'*12} {'-'*12} {'-'*10}")
    print(f"{'Finetune':<40} {ft_mean:<12.4f} {'1.0x':<12} {'-':<10}")
    print(f"{'Pretrain (iter3: with linebreaks)':<40} {pt3_mean:<12.4f} {ft_mean/pt3_mean:<12.1f}x {ft_wins_iter3:<10}")
    print(f"{'Pretrain (iter4: no linebreaks)':<40} {pt4_mean:<12.4f} {ft_mean/pt4_mean:<12.1f}x {ft_wins_iter4:<10}")

    # Improvement
    improvement = (pt4_mean - pt3_mean) / pt3_mean * 100

    print(f"\n{'='*70}")
    print(f"EFFECT OF REMOVING LINEBREAKS")
    print(f"{'='*70}")
    print(f"Pretrain influence change: {improvement:+.1f}%")
    print(f"  Iter3 (with \\n): {pt3_mean:.4f}")
    print(f"  Iter4 (no \\n):   {pt4_mean:.4f}")

    if improvement > 0:
        print(f"\n✓ Removing linebreaks IMPROVED pretrain influence by {improvement:.1f}%")
    else:
        print(f"\n✗ Removing linebreaks did NOT improve pretrain influence ({improvement:.1f}%)")

    # Distribution comparison
    print(f"\n{'='*70}")
    print(f"DISTRIBUTION COMPARISON")
    print(f"{'='*70}")
    print(f"{'Metric':<15} {'FT':<12} {'PT(iter3)':<12} {'PT(iter4)':<12}")
    print(f"{'-'*15} {'-'*12} {'-'*12} {'-'*12}")
    print(f"{'Max':<15} {ft_max.max().item():<12.4f} {pt3_max.max().item():<12.4f} {pt4_max.max().item():<12.4f}")
    print(f"{'Min':<15} {ft_max.min().item():<12.4f} {pt3_max.min().item():<12.4f} {pt4_max.min().item():<12.4f}")
    print(f"{'Mean':<15} {ft_mean:<12.4f} {pt3_mean:<12.4f} {pt4_mean:<12.4f}")
    print(f"{'Median':<15} {ft_max.median().item():<12.4f} {pt3_max.median().item():<12.4f} {pt4_max.median().item():<12.4f}")
    print(f"{'Std':<15} {ft_max.std().item():<12.4f} {pt3_max.std().item():<12.4f} {pt4_max.std().item():<12.4f}")

    # Save results
    results = {
        "finetune_mean": ft_mean,
        "pretrain_iter3_mean": pt3_mean,
        "pretrain_iter4_mean": pt4_mean,
        "ft_pt_ratio_iter3": ft_mean / pt3_mean,
        "ft_pt_ratio_iter4": ft_mean / pt4_mean,
        "improvement_pct": improvement,
        "ft_wins_iter3": ft_wins_iter3,
        "ft_wins_iter4": ft_wins_iter4,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "no_linebreaks_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
