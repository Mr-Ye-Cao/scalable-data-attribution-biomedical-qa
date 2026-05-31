#!/usr/bin/env python3
"""
Analyze Label-Matched Controlled Experiment

Compare pretrain influence with label-matched outputs vs original (all "maybe").
"""

import json
import torch
from pathlib import Path

BASE_DIR = Path("/lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune")

# Paths
PREDICTIONS_DIR = BASE_DIR / "eval_output_results"
ALIGNED_GRADS = BASE_DIR / "results/rapidin_aligned_500/grads"  # finetune & test
CONTROLLED_GRADS = BASE_DIR / "results/controlled_experiment/grads"  # pretrain (all maybe)
MATCHED_GRADS = BASE_DIR / "results/controlled_experiment/grads_matched"  # pretrain (label matched)

OUTPUT_DIR = BASE_DIR / "results/controlled_experiment/analysis"


def load_predictions(model_name):
    pred_files = {
        "baseline": "baseline_predictions_full.json",
        "ckpt32": "epoch1_checkpoint-32_predictions_full.json",
        "ckpt64": "epoch2_checkpoint-64_predictions_full.json",
    }
    with open(PREDICTIONS_DIR / pred_files[model_name]) as f:
        return json.load(f)


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
    print("LABEL-MATCHED vs ALL-MAYBE PRETRAIN COMPARISON")
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
        pretrain_maybe = load_gradients(CONTROLLED_GRADS / model_name / "pretrain")
        pretrain_matched = load_gradients(MATCHED_GRADS / model_name / "pretrain")

        if any(g is None for g in [test_grads, finetune_grads, pretrain_maybe, pretrain_matched]):
            print(f"Missing gradients for {model_name}")
            continue

        print(f"  Test: {test_grads.shape}")
        print(f"  Finetune: {finetune_grads.shape}")
        print(f"  Pretrain (maybe): {pretrain_maybe.shape}")
        print(f"  Pretrain (matched): {pretrain_matched.shape}")

        # Compute influence
        print("\nComputing influence scores...")
        ft_infl = compute_influence(test_grads, finetune_grads)
        pt_maybe_infl = compute_influence(test_grads, pretrain_maybe)
        pt_matched_infl = compute_influence(test_grads, pretrain_matched)

        # Stats
        ft_max = ft_infl.max(dim=1).values
        pt_maybe_max = pt_maybe_infl.max(dim=1).values
        pt_matched_max = pt_matched_infl.max(dim=1).values

        ft_mean = float(ft_max.mean())
        pt_maybe_mean = float(pt_maybe_max.mean())
        pt_matched_mean = float(pt_matched_max.mean())

        # Compare
        print(f"\n{'='*60}")
        print(f"RESULTS: {model_name.upper()}")
        print(f"{'='*60}")
        print(f"{'Dataset':<25} {'Max Mean':<12} {'FT/PT Ratio':<12}")
        print(f"{'-'*25} {'-'*12} {'-'*12}")
        print(f"{'Finetune':<25} {ft_mean:<12.4f} {'1.0x':<12}")
        print(f"{'Pretrain (all maybe)':<25} {pt_maybe_mean:<12.4f} {ft_mean/pt_maybe_mean:<12.1f}x")
        print(f"{'Pretrain (label-matched)':<25} {pt_matched_mean:<12.4f} {ft_mean/pt_matched_mean:<12.1f}x")

        # Improvement
        improvement = (pt_matched_mean - pt_maybe_mean) / pt_maybe_mean * 100
        print(f"\nLabel-matching improvement: {improvement:+.1f}%")

        # FT wins comparison
        ft_wins_maybe = (ft_max > pt_maybe_max).sum().item()
        ft_wins_matched = (ft_max > pt_matched_max).sum().item()
        print(f"FT wins (vs maybe):   {ft_wins_maybe}/500")
        print(f"FT wins (vs matched): {ft_wins_matched}/500")

        results[model_name] = {
            "finetune_mean": ft_mean,
            "pretrain_maybe_mean": pt_maybe_mean,
            "pretrain_matched_mean": pt_matched_mean,
            "ft_pt_ratio_maybe": ft_mean / pt_maybe_mean,
            "ft_pt_ratio_matched": ft_mean / pt_matched_mean,
            "improvement_pct": improvement,
            "ft_wins_maybe": ft_wins_maybe,
            "ft_wins_matched": ft_wins_matched,
        }

    # Summary comparison
    print(f"\n\n{'#'*80}")
    print("# SUMMARY: LABEL-MATCHING EFFECT")
    print(f"{'#'*80}")

    print(f"\n{'Model':<10} {'PT (maybe)':<12} {'PT (matched)':<14} {'Change':<10} {'FT/PT (m)':<12} {'FT/PT (M)':<12}")
    print(f"{'-'*10} {'-'*12} {'-'*14} {'-'*10} {'-'*12} {'-'*12}")

    for model_name in ["baseline", "ckpt32", "ckpt64"]:
        if model_name not in results:
            continue
        r = results[model_name]
        print(f"{model_name:<10} {r['pretrain_maybe_mean']:<12.4f} {r['pretrain_matched_mean']:<14.4f} {r['improvement_pct']:>+8.1f}% {r['ft_pt_ratio_maybe']:<12.1f}x {r['ft_pt_ratio_matched']:<12.1f}x")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "label_matched_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
