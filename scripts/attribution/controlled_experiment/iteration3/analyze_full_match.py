#!/usr/bin/env python3
"""
Analyze Full-Match Controlled Experiment (Iteration 3)

Compare pretrain influence across all three iterations:
- Iteration 1: All "maybe" labels
- Iteration 2: Label-matched (yes/no/maybe from source query)
- Iteration 3: Full match (entire output including Long Answer)
"""

import json
import torch
from pathlib import Path

BASE_DIR = Path("/lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune")

# Paths
ALIGNED_GRADS = BASE_DIR / "results/rapidin_aligned_500/grads"  # finetune & test
ITER1_GRADS = BASE_DIR / "results/controlled_experiment/grads"  # pretrain (all maybe)
ITER2_GRADS = BASE_DIR / "results/controlled_experiment/grads_matched"  # pretrain (label matched)
ITER3_GRADS = BASE_DIR / "results/controlled_experiment/iteration3/grads"  # pretrain (full match)

OUTPUT_DIR = BASE_DIR / "results/controlled_experiment/iteration3/analysis"


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
    print("ITERATION 3: FULL-MATCH vs LABEL-MATCHED vs ALL-MAYBE")
    print("=" * 80)

    results = {}

    for model_name in ["baseline", "ckpt32", "ckpt64"]:
        print(f"\n{'#'*80}")
        print(f"# {model_name.upper()}")
        print(f"{'#'*80}")

        # Load gradients
        print(f"\nLoading gradients...")
        test_grads = load_gradients(ALIGNED_GRADS / model_name / "test")
        finetune_grads = load_gradients(ALIGNED_GRADS / model_name / "finetune")
        pt_iter1 = load_gradients(ITER1_GRADS / model_name / "pretrain")
        pt_iter2 = load_gradients(ITER2_GRADS / model_name / "pretrain")
        pt_iter3 = load_gradients(ITER3_GRADS / model_name / "pretrain")

        missing = []
        if test_grads is None: missing.append("test")
        if finetune_grads is None: missing.append("finetune")
        if pt_iter1 is None: missing.append("pretrain_iter1")
        if pt_iter2 is None: missing.append("pretrain_iter2")
        if pt_iter3 is None: missing.append("pretrain_iter3")

        if missing:
            print(f"Missing gradients: {missing}")
            continue

        print(f"  Test: {test_grads.shape}")
        print(f"  Finetune: {finetune_grads.shape}")
        print(f"  Pretrain (iter1/all-maybe): {pt_iter1.shape}")
        print(f"  Pretrain (iter2/label-matched): {pt_iter2.shape}")
        print(f"  Pretrain (iter3/full-match): {pt_iter3.shape}")

        # Compute influence
        print("\nComputing influence scores...")
        ft_infl = compute_influence(test_grads, finetune_grads)
        pt1_infl = compute_influence(test_grads, pt_iter1)
        pt2_infl = compute_influence(test_grads, pt_iter2)
        pt3_infl = compute_influence(test_grads, pt_iter3)

        # Stats - max influence per test sample
        ft_max = ft_infl.max(dim=1).values
        pt1_max = pt1_infl.max(dim=1).values
        pt2_max = pt2_infl.max(dim=1).values
        pt3_max = pt3_infl.max(dim=1).values

        ft_mean = float(ft_max.mean())
        pt1_mean = float(pt1_max.mean())
        pt2_mean = float(pt2_max.mean())
        pt3_mean = float(pt3_max.mean())

        # FT wins comparison
        ft_wins_iter1 = (ft_max > pt1_max).sum().item()
        ft_wins_iter2 = (ft_max > pt2_max).sum().item()
        ft_wins_iter3 = (ft_max > pt3_max).sum().item()

        # Compare
        print(f"\n{'='*70}")
        print(f"RESULTS: {model_name.upper()}")
        print(f"{'='*70}")
        print(f"{'Dataset':<35} {'Max Mean':<12} {'FT/PT Ratio':<12} {'FT Wins':<10}")
        print(f"{'-'*35} {'-'*12} {'-'*12} {'-'*10}")
        print(f"{'Finetune':<35} {ft_mean:<12.4f} {'1.0x':<12} {'-':<10}")
        print(f"{'Pretrain (iter1: all-maybe)':<35} {pt1_mean:<12.4f} {ft_mean/pt1_mean:<12.1f}x {ft_wins_iter1:<10}")
        print(f"{'Pretrain (iter2: label-matched)':<35} {pt2_mean:<12.4f} {ft_mean/pt2_mean:<12.1f}x {ft_wins_iter2:<10}")
        print(f"{'Pretrain (iter3: full-match)':<35} {pt3_mean:<12.4f} {ft_mean/pt3_mean:<12.1f}x {ft_wins_iter3:<10}")

        # Improvement over iterations
        imp_iter2 = (pt2_mean - pt1_mean) / pt1_mean * 100
        imp_iter3_vs_1 = (pt3_mean - pt1_mean) / pt1_mean * 100
        imp_iter3_vs_2 = (pt3_mean - pt2_mean) / pt2_mean * 100

        print(f"\nPretrain Influence Improvements:")
        print(f"  Iter2 vs Iter1 (label matching):    {imp_iter2:+.1f}%")
        print(f"  Iter3 vs Iter1 (full match):        {imp_iter3_vs_1:+.1f}%")
        print(f"  Iter3 vs Iter2 (add long answer):   {imp_iter3_vs_2:+.1f}%")

        results[model_name] = {
            "finetune_mean": ft_mean,
            "pretrain_iter1_mean": pt1_mean,
            "pretrain_iter2_mean": pt2_mean,
            "pretrain_iter3_mean": pt3_mean,
            "ft_pt_ratio_iter1": ft_mean / pt1_mean,
            "ft_pt_ratio_iter2": ft_mean / pt2_mean,
            "ft_pt_ratio_iter3": ft_mean / pt3_mean,
            "improvement_iter2_vs_iter1": imp_iter2,
            "improvement_iter3_vs_iter1": imp_iter3_vs_1,
            "improvement_iter3_vs_iter2": imp_iter3_vs_2,
            "ft_wins_iter1": ft_wins_iter1,
            "ft_wins_iter2": ft_wins_iter2,
            "ft_wins_iter3": ft_wins_iter3,
        }

    # Summary comparison
    print(f"\n\n{'#'*80}")
    print("# SUMMARY: PRETRAIN INFLUENCE ACROSS ITERATIONS")
    print(f"{'#'*80}")

    print(f"\n{'Model':<10} {'PT(iter1)':<12} {'PT(iter2)':<12} {'PT(iter3)':<12} {'FT':<12}")
    print(f"{'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    for model_name in ["baseline", "ckpt32", "ckpt64"]:
        if model_name not in results:
            continue
        r = results[model_name]
        print(f"{model_name:<10} {r['pretrain_iter1_mean']:<12.4f} {r['pretrain_iter2_mean']:<12.4f} {r['pretrain_iter3_mean']:<12.4f} {r['finetune_mean']:<12.4f}")

    print(f"\n{'Model':<10} {'FT/PT(i1)':<12} {'FT/PT(i2)':<12} {'FT/PT(i3)':<12} {'i3 vs i1':<12}")
    print(f"{'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    for model_name in ["baseline", "ckpt32", "ckpt64"]:
        if model_name not in results:
            continue
        r = results[model_name]
        print(f"{model_name:<10} {r['ft_pt_ratio_iter1']:<12.1f}x {r['ft_pt_ratio_iter2']:<12.1f}x {r['ft_pt_ratio_iter3']:<12.1f}x {r['improvement_iter3_vs_iter1']:>+10.1f}%")

    print(f"\n{'Model':<10} {'FT wins(i1)':<14} {'FT wins(i2)':<14} {'FT wins(i3)':<14}")
    print(f"{'-'*10} {'-'*14} {'-'*14} {'-'*14}")

    for model_name in ["baseline", "ckpt32", "ckpt64"]:
        if model_name not in results:
            continue
        r = results[model_name]
        print(f"{model_name:<10} {r['ft_wins_iter1']}/500       {r['ft_wins_iter2']}/500       {r['ft_wins_iter3']}/500")

    # Key insight
    print(f"\n\n{'#'*80}")
    print("# KEY INSIGHT")
    print(f"{'#'*80}")
    print("""
If Iteration 3 (full match) shows significantly higher pretrain influence
than Iteration 2 (label-matched), it means the Long Answer content matters
for gradient similarity - not just the final decision label.

If Iteration 3 still shows finetune >> pretrain, then the CONTEXT (instruction)
is the key differentiating factor, since the OUTPUT is now identical.
""")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "full_match_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
