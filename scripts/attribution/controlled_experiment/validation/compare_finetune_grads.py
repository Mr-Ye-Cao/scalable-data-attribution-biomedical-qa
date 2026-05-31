#!/usr/bin/env python3
"""
Validate finetune gradients by comparing original vs recomputed.
"""

import torch
from pathlib import Path

BASE_DIR = Path("/lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune")

ORIGINAL_FT_GRADS = BASE_DIR / "results/rapidin_aligned_500/grads/baseline/finetune"
RECOMPUTED_FT_GRADS = BASE_DIR / "results/validation/grads/baseline/finetune"
TEST_GRADS = BASE_DIR / "results/rapidin_aligned_500/grads/baseline/test"


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
    print("=" * 70)
    print("VALIDATION: Compare Original vs Recomputed Finetune Gradients")
    print("=" * 70)

    # Load gradients
    print("\nLoading gradients...")
    test_grads = load_gradients(TEST_GRADS)
    original_ft = load_gradients(ORIGINAL_FT_GRADS)
    recomputed_ft = load_gradients(RECOMPUTED_FT_GRADS)

    if test_grads is None:
        print("ERROR: Test gradients not found")
        return
    if original_ft is None:
        print("ERROR: Original finetune gradients not found")
        return
    if recomputed_ft is None:
        print("ERROR: Recomputed finetune gradients not found")
        return

    print(f"  Test: {test_grads.shape}")
    print(f"  Original FT: {original_ft.shape}")
    print(f"  Recomputed FT: {recomputed_ft.shape}")

    # Compare original vs recomputed gradients directly
    print("\n" + "=" * 70)
    print("GRADIENT COMPARISON: Original vs Recomputed")
    print("=" * 70)

    # Check if they're identical
    diff = (original_ft - recomputed_ft).norm(dim=1)
    print(f"\nL2 distance per sample:")
    print(f"  Mean: {diff.mean():.4f}")
    print(f"  Max:  {diff.max():.4f}")
    print(f"  Min:  {diff.min():.4f}")

    # Cosine similarity between original and recomputed
    orig_norm = original_ft / (original_ft.norm(dim=1, keepdim=True) + 1e-8)
    recomp_norm = recomputed_ft / (recomputed_ft.norm(dim=1, keepdim=True) + 1e-8)
    cos_sim = (orig_norm * recomp_norm).sum(dim=1)
    print(f"\nCosine similarity (should be ~1.0 if identical):")
    print(f"  Mean: {cos_sim.mean():.6f}")
    print(f"  Min:  {cos_sim.min():.6f}")
    print(f"  Max:  {cos_sim.max():.6f}")

    # Compute influence scores
    print("\n" + "=" * 70)
    print("INFLUENCE COMPARISON")
    print("=" * 70)

    print("\nComputing influence scores...")
    orig_infl = compute_influence(test_grads, original_ft)
    recomp_infl = compute_influence(test_grads, recomputed_ft)

    # Per-test max
    orig_max = orig_infl.max(dim=1).values
    recomp_max = recomp_infl.max(dim=1).values

    print(f"\nPer-test MAX influence:")
    print(f"{'Metric':<20} {'Original':<15} {'Recomputed':<15}")
    print(f"{'-'*20} {'-'*15} {'-'*15}")
    print(f"{'Mean':<20} {orig_max.mean():.4f}         {recomp_max.mean():.4f}")
    print(f"{'Std':<20} {orig_max.std():.4f}         {recomp_max.std():.4f}")
    print(f"{'Min':<20} {orig_max.min():.4f}         {recomp_max.min():.4f}")
    print(f"{'Max':<20} {orig_max.max():.4f}         {recomp_max.max():.4f}")

    # Difference in influence scores
    infl_diff = (orig_max - recomp_max).abs()
    print(f"\nInfluence score difference:")
    print(f"  Mean abs diff: {infl_diff.mean():.6f}")
    print(f"  Max abs diff:  {infl_diff.max():.6f}")

    # Correlation
    correlation = torch.corrcoef(torch.stack([orig_max, recomp_max]))[0, 1]
    print(f"\nCorrelation between original and recomputed: {correlation:.6f}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if cos_sim.mean() > 0.99:
        print("✓ Gradients are essentially IDENTICAL (cosine sim > 0.99)")
    elif cos_sim.mean() > 0.95:
        print("~ Gradients are very similar (cosine sim > 0.95)")
    else:
        print("✗ Gradients are DIFFERENT")

    if infl_diff.mean() < 0.01:
        print("✓ Influence scores are essentially IDENTICAL (diff < 0.01)")
    elif infl_diff.mean() < 0.05:
        print("~ Influence scores are very similar (diff < 0.05)")
    else:
        print("✗ Influence scores are DIFFERENT")

    print(f"\nFinal answer: Finetune influence score = {recomp_max.mean():.4f}")


if __name__ == "__main__":
    main()
