#!/usr/bin/env python3
"""
Analyze influence scores by correctness category.

Categories:
1. Newly Solved: Baseline wrong -> Checkpoint correct
2. Broken: Baseline correct -> Checkpoint wrong
3. Consistently Correct: Baseline correct -> Checkpoint correct
4. Consistently Wrong: Baseline wrong -> Checkpoint wrong

For each category, compare pretrain vs finetune influence distributions.
"""

import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

BASE_DIR = Path("/lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune")

# Paths
PREDICTIONS_DIR = BASE_DIR / "eval_output_results"
ALIGNED_GRADS = BASE_DIR / "results/rapidin_aligned_500/grads"
ORIGINAL_GRADS = BASE_DIR / "results/rapidin_original/grads"


def load_predictions(model_name):
    """Load predictions for a model."""
    pred_files = {
        "baseline": "baseline_predictions_full.json",
        "ckpt32": "epoch1_checkpoint-32_predictions_full.json",
        "ckpt64": "epoch2_checkpoint-64_predictions_full.json",
        "ckpt96": "epoch3_final_predictions_full.json",
    }

    pred_path = PREDICTIONS_DIR / pred_files[model_name]
    with open(pred_path) as f:
        return json.load(f)


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
    for f in files:
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


def categorize_queries(baseline_preds, ckpt_preds):
    """
    Categorize queries based on correctness transition.

    Returns dict with category -> list of (index, pmid) tuples
    """
    # Get PMIDs in order
    pmids = list(baseline_preds.keys())[:500]  # We use first 500 for gradients

    categories = {
        "newly_solved": [],      # Baseline wrong -> Ckpt correct
        "broken": [],            # Baseline correct -> Ckpt wrong
        "consistently_correct": [],  # Both correct
        "consistently_wrong": [],    # Both wrong
    }

    for idx, pmid in enumerate(pmids):
        baseline_correct = baseline_preds[pmid]["correct"]
        ckpt_correct = ckpt_preds[pmid]["correct"]

        if not baseline_correct and ckpt_correct:
            categories["newly_solved"].append((idx, pmid))
        elif baseline_correct and not ckpt_correct:
            categories["broken"].append((idx, pmid))
        elif baseline_correct and ckpt_correct:
            categories["consistently_correct"].append((idx, pmid))
        else:
            categories["consistently_wrong"].append((idx, pmid))

    return categories


def analyze_influence_by_category(categories, finetune_infl, pretrain_infl):
    """Analyze influence statistics for each category."""
    results = {}

    for cat_name, queries in categories.items():
        if len(queries) == 0:
            results[cat_name] = {"count": 0}
            continue

        indices = [q[0] for q in queries]

        # Get influence scores for this category
        ft_scores = finetune_infl[indices]  # [num_queries, num_train]
        pt_scores = pretrain_infl[indices]  # [num_queries, num_train]

        # Compute statistics
        ft_max_per_query = ft_scores.max(dim=1).values
        pt_max_per_query = pt_scores.max(dim=1).values

        ft_mean_per_query = ft_scores.mean(dim=1)
        pt_mean_per_query = pt_scores.mean(dim=1)

        results[cat_name] = {
            "count": len(queries),
            "finetune": {
                "mean": float(ft_scores.mean()),
                "std": float(ft_scores.std()),
                "max": float(ft_scores.max()),
                "max_per_query_mean": float(ft_max_per_query.mean()),
                "max_per_query_std": float(ft_max_per_query.std()),
                "per_query_means": ft_mean_per_query.tolist(),
                "per_query_maxes": ft_max_per_query.tolist(),
            },
            "pretrain": {
                "mean": float(pt_scores.mean()),
                "std": float(pt_scores.std()),
                "max": float(pt_scores.max()),
                "max_per_query_mean": float(pt_max_per_query.mean()),
                "max_per_query_std": float(pt_max_per_query.std()),
                "per_query_means": pt_mean_per_query.tolist(),
                "per_query_maxes": pt_max_per_query.tolist(),
            },
            "pmids": [q[1] for q in queries],
        }

        # FT/PT ratio per query
        ft_wins = (ft_max_per_query > pt_max_per_query).sum().item()
        results[cat_name]["ft_wins"] = ft_wins
        results[cat_name]["ft_ratio"] = float(ft_max_per_query.mean() / (pt_max_per_query.mean() + 1e-8))

    return results


def print_category_analysis(model_name, results):
    """Print analysis results for a model."""
    print(f"\n{'='*70}")
    print(f"ANALYSIS: Baseline vs {model_name.upper()}")
    print(f"{'='*70}")

    print(f"\n{'Category':<22} {'Count':<8} {'FT Mean':<12} {'PT Mean':<12} {'FT/PT':<10} {'FT Wins':<10}")
    print(f"{'-'*22} {'-'*8} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")

    for cat in ["newly_solved", "broken", "consistently_correct", "consistently_wrong"]:
        r = results[cat]
        if r["count"] == 0:
            print(f"{cat:<22} {0:<8}")
            continue

        ft_mean = r["finetune"]["max_per_query_mean"]
        pt_mean = r["pretrain"]["max_per_query_mean"]
        ratio = ft_mean / (pt_mean + 1e-8)
        ft_wins = r["ft_wins"]
        count = r["count"]

        print(f"{cat:<22} {count:<8} {ft_mean:<12.4f} {pt_mean:<12.4f} {ratio:<10.1f}x {ft_wins}/{count}")


def main():
    print("=" * 70)
    print("Fine-Grained Influence Analysis by Correctness Category")
    print("=" * 70)

    # Load predictions
    print("\nLoading predictions...")
    baseline_preds = load_predictions("baseline")
    ckpt32_preds = load_predictions("ckpt32")
    ckpt64_preds = load_predictions("ckpt64")
    ckpt96_preds = load_predictions("ckpt96")

    # Load baseline gradients first (we'll need them for comparison)
    print("\nLoading baseline gradients...")
    baseline_test_grads = load_gradients(ALIGNED_GRADS / "baseline" / "test")
    baseline_finetune_grads = load_gradients(ALIGNED_GRADS / "baseline" / "finetune", max_samples=500)
    baseline_pretrain_grads = load_gradients(ORIGINAL_GRADS / "baseline" / "pretrain", max_samples=500)

    print(f"  Baseline test shape: {baseline_test_grads.shape}")
    print(f"  Baseline finetune shape: {baseline_finetune_grads.shape}")
    print(f"  Baseline pretrain shape: {baseline_pretrain_grads.shape}")

    # Compute baseline influence
    print("\nComputing baseline influence...")
    baseline_ft_infl = compute_influence(baseline_test_grads, baseline_finetune_grads)
    baseline_pt_infl = compute_influence(baseline_test_grads, baseline_pretrain_grads)

    all_results = {}

    for model_name, ckpt_preds in [("ckpt32", ckpt32_preds), ("ckpt64", ckpt64_preds), ("ckpt96", ckpt96_preds)]:
        print(f"\n\n{'#'*70}")
        print(f"# Analyzing {model_name.upper()}")
        print(f"{'#'*70}")

        # Categorize queries
        categories = categorize_queries(baseline_preds, ckpt_preds)

        print(f"\nCategory counts:")
        for cat, queries in categories.items():
            print(f"  {cat}: {len(queries)}")

        # Load gradients for this model
        print(f"\nLoading {model_name} gradients...")
        test_grads = load_gradients(ALIGNED_GRADS / model_name / "test")
        finetune_grads = load_gradients(ALIGNED_GRADS / model_name / "finetune", max_samples=500)
        pretrain_grads = load_gradients(ORIGINAL_GRADS / model_name / "pretrain", max_samples=500)

        if test_grads is None or finetune_grads is None or pretrain_grads is None:
            print(f"Skipping {model_name} - missing gradients")
            continue

        print(f"  Test shape: {test_grads.shape}")
        print(f"  Finetune shape: {finetune_grads.shape}")
        print(f"  Pretrain shape: {pretrain_grads.shape}")

        # Compute influence scores
        print("\nComputing influence scores...")
        finetune_infl = compute_influence(test_grads, finetune_grads)
        pretrain_infl = compute_influence(test_grads, pretrain_grads)

        # Analyze by category - for both checkpoint and baseline
        results = analyze_influence_by_category(categories, finetune_infl, pretrain_infl)
        baseline_results = analyze_influence_by_category(categories, baseline_ft_infl, baseline_pt_infl)

        # Print results
        print_category_analysis(model_name, results)

        # Print comparison with baseline
        print(f"\n{'-'*70}")
        print(f"CHANGE FROM BASELINE TO {model_name.upper()}")
        print(f"{'-'*70}")
        print(f"{'Category':<22} {'BL FT':<10} {'Ckpt FT':<10} {'FT Chg':<12} {'BL PT':<10} {'Ckpt PT':<10} {'PT Chg':<12}")
        print(f"{'-'*22} {'-'*10} {'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*12}")

        for cat in ["newly_solved", "broken", "consistently_correct", "consistently_wrong"]:
            r = results[cat]
            br = baseline_results[cat]
            if r["count"] == 0:
                print(f"{cat:<22} (no samples)")
                continue

            bl_ft = br["finetune"]["max_per_query_mean"]
            ckpt_ft = r["finetune"]["max_per_query_mean"]
            ft_change = (ckpt_ft - bl_ft) / bl_ft * 100 if bl_ft > 0 else 0

            bl_pt = br["pretrain"]["max_per_query_mean"]
            ckpt_pt = r["pretrain"]["max_per_query_mean"]
            pt_change = (ckpt_pt - bl_pt) / bl_pt * 100 if bl_pt > 0 else 0

            print(f"{cat:<22} {bl_ft:<10.4f} {ckpt_ft:<10.4f} {ft_change:>+10.1f}% {bl_pt:<10.4f} {ckpt_pt:<10.4f} {pt_change:>+10.1f}%")

        all_results[model_name] = {
            "checkpoint": results,
            "baseline": baseline_results,
        }

    # Cross-model comparison with baseline changes
    print(f"\n\n{'#'*70}")
    print("# CROSS-MODEL COMPARISON: INFLUENCE CHANGES FROM BASELINE")
    print(f"{'#'*70}")

    for cat in ["newly_solved", "broken", "consistently_correct", "consistently_wrong"]:
        print(f"\n## {cat.upper().replace('_', ' ')}")
        print(f"{'Model':<10} {'Count':<8} {'BL FT':<10} {'Ckpt FT':<10} {'FT Chg':<12} {'BL PT':<10} {'Ckpt PT':<10} {'PT Chg':<12}")
        print(f"{'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*12}")

        for model_name in ["ckpt32", "ckpt64", "ckpt96"]:
            if model_name not in all_results:
                continue
            r = all_results[model_name]["checkpoint"][cat]
            br = all_results[model_name]["baseline"][cat]
            if r["count"] == 0:
                print(f"{model_name:<10} {0:<8}")
                continue

            bl_ft = br["finetune"]["max_per_query_mean"]
            ckpt_ft = r["finetune"]["max_per_query_mean"]
            ft_change = (ckpt_ft - bl_ft) / bl_ft * 100 if bl_ft > 0 else 0

            bl_pt = br["pretrain"]["max_per_query_mean"]
            ckpt_pt = r["pretrain"]["max_per_query_mean"]
            pt_change = (ckpt_pt - bl_pt) / bl_pt * 100 if bl_pt > 0 else 0

            print(f"{model_name:<10} {r['count']:<8} {bl_ft:<10.4f} {ckpt_ft:<10.4f} {ft_change:>+10.1f}% {bl_pt:<10.4f} {ckpt_pt:<10.4f} {pt_change:>+10.1f}%")

    # Save results
    output_path = BASE_DIR / "results/rapidin_aligned_500/correctness_category_analysis.json"

    # Convert for JSON serialization
    def clean_results(results_dict):
        cleaned = {}
        for cat, r in results_dict.items():
            cleaned[cat] = {
                k: v for k, v in r.items()
                if k not in ["per_query_means", "per_query_maxes", "pmids"]
            }
            if "finetune" in r:
                cleaned[cat]["finetune"] = {
                    k: v for k, v in r["finetune"].items()
                    if k not in ["per_query_means", "per_query_maxes"]
                }
            if "pretrain" in r:
                cleaned[cat]["pretrain"] = {
                    k: v for k, v in r["pretrain"].items()
                    if k not in ["per_query_means", "per_query_maxes"]
                }
        return cleaned

    serializable_results = {}
    for model_name, model_data in all_results.items():
        serializable_results[model_name] = {
            "checkpoint": clean_results(model_data["checkpoint"]),
            "baseline": clean_results(model_data["baseline"]),
        }

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\n\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
