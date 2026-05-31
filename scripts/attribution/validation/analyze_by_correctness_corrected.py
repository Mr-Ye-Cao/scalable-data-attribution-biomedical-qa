#!/usr/bin/env python3
"""
Analyze influence scores by correctness category - CORRECTED VERSION.

Uses the correct pretrain gradients (noformat and iter1) computed with
run_rapidin_original.py instead of the buggy MP_main.py.

Categories:
1. Newly Solved: Baseline wrong -> Checkpoint correct
2. Broken: Baseline correct -> Checkpoint wrong
3. Consistently Correct: Baseline correct -> Checkpoint correct
4. Consistently Wrong: Baseline wrong -> Checkpoint wrong
"""

import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("/lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune")

# Paths
PREDICTIONS_DIR = BASE_DIR / "eval_output_results"
ALIGNED_GRADS = BASE_DIR / "results/rapidin_aligned_500/grads"
PT_GRADS_DIR = BASE_DIR / "results/validation/grads"

OUTPUT_DIR = BASE_DIR / "results/validation/analysis"

MODELS = ["baseline", "ckpt32", "ckpt64"]
DATASETS = ["finetune", "noformat", "iter1"]


def load_predictions(model_name):
    """Load predictions for a model."""
    pred_files = {
        "baseline": "baseline_predictions_full.json",
        "ckpt32": "epoch1_checkpoint-32_predictions_full.json",
        "ckpt64": "epoch2_checkpoint-64_predictions_full.json",
    }
    pred_path = PREDICTIONS_DIR / pred_files[model_name]
    with open(pred_path) as f:
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


def categorize_queries(baseline_preds, ckpt_preds):
    """Categorize queries based on correctness transition."""
    pmids = list(baseline_preds.keys())[:500]

    categories = {
        "newly_solved": [],
        "broken": [],
        "consistently_correct": [],
        "consistently_wrong": [],
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


def analyze_category(categories, influence_dict):
    """Analyze influence for each category across datasets."""
    results = {}

    for cat_name, queries in categories.items():
        if len(queries) == 0:
            results[cat_name] = {"count": 0}
            continue

        indices = [q[0] for q in queries]
        results[cat_name] = {"count": len(queries)}

        for dataset, infl_matrix in influence_dict.items():
            if infl_matrix is None:
                continue

            scores = infl_matrix[indices]
            max_per_query = scores.max(dim=1).values

            results[cat_name][dataset] = {
                "mean": float(max_per_query.mean()),
                "std": float(max_per_query.std()),
                "min": float(max_per_query.min()),
                "max": float(max_per_query.max()),
            }

        # Compute FT wins vs each pretrain dataset
        if "finetune" in results[cat_name]:
            ft_scores = influence_dict["finetune"][indices].max(dim=1).values
            for pt_dataset in ["noformat", "iter1"]:
                if pt_dataset in influence_dict and influence_dict[pt_dataset] is not None:
                    pt_scores = influence_dict[pt_dataset][indices].max(dim=1).values
                    ft_wins = int((ft_scores > pt_scores).sum().item())
                    pt_wins = int((pt_scores > ft_scores).sum().item())
                    results[cat_name][f"ft_wins_vs_{pt_dataset}"] = ft_wins
                    results[cat_name][f"pt_wins_vs_{pt_dataset}"] = pt_wins

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CORRECTNESS CATEGORY ANALYSIS - CORRECTED VERSION")
    print(f"Time: {datetime.now()}")
    print("=" * 80)

    # Load predictions
    print("\nLoading predictions...")
    baseline_preds = load_predictions("baseline")
    ckpt32_preds = load_predictions("ckpt32")
    ckpt64_preds = load_predictions("ckpt64")

    all_results = {}

    for model in MODELS:
        print(f"\n{'#' * 60}")
        print(f"# MODEL: {model.upper()}")
        print(f"{'#' * 60}")

        # Load gradients
        test_grads = load_gradients(ALIGNED_GRADS / model / "test")
        if test_grads is None:
            print(f"Missing test gradients for {model}")
            continue

        # Load all training gradients
        influence_dict = {}

        # Finetune
        ft_grads = load_gradients(ALIGNED_GRADS / model / "finetune")
        if ft_grads is not None:
            influence_dict["finetune"] = compute_influence(test_grads, ft_grads)
            print(f"Finetune gradients: {ft_grads.shape}")

        # NoFormat
        nf_grads = load_gradients(PT_GRADS_DIR / model / "noformat")
        if nf_grads is not None:
            influence_dict["noformat"] = compute_influence(test_grads, nf_grads)
            print(f"NoFormat gradients: {nf_grads.shape}")

        # Iter1
        iter1_grads = load_gradients(PT_GRADS_DIR / model / "iter1")
        if iter1_grads is None:
            iter1_grads = load_gradients(PT_GRADS_DIR / model / "pretrain_iter1")
        if iter1_grads is not None:
            influence_dict["iter1"] = compute_influence(test_grads, iter1_grads)
            print(f"Iter1 gradients: {iter1_grads.shape}")

        model_results = {}

        # Analyze for each checkpoint comparison
        for ckpt_name, ckpt_preds in [("ckpt32", ckpt32_preds), ("ckpt64", ckpt64_preds)]:
            if model == "baseline":
                # For baseline, use categories from this checkpoint
                categories = categorize_queries(baseline_preds, ckpt_preds)
            else:
                # For checkpoints, use their own categories
                if model == ckpt_name:
                    categories = categorize_queries(baseline_preds, ckpt_preds)
                else:
                    continue

            print(f"\nCategories (Baseline -> {ckpt_name}):")
            for cat, queries in categories.items():
                print(f"  {cat}: {len(queries)}")

            results = analyze_category(categories, influence_dict)
            model_results[f"vs_{ckpt_name}"] = results

        all_results[model] = model_results

    # Print summary tables
    print("\n" + "=" * 80)
    print("SUMMARY TABLES")
    print("=" * 80)

    for comparison in ["vs_ckpt32", "vs_ckpt64"]:
        ckpt_name = comparison.replace("vs_", "")
        print(f"\n\n{'#' * 60}")
        print(f"# Baseline -> {ckpt_name.upper()}")
        print(f"{'#' * 60}")

        # Print table for each model
        for model in MODELS:
            if model not in all_results or comparison not in all_results[model]:
                continue

            results = all_results[model][comparison]

            print(f"\n## {model.upper()}")
            print(f"{'Category':<22} {'Count':<7} {'FT':<10} {'NoFmt':<10} {'Iter1':<10} {'FT/NF':<8} {'FT/I1':<8} {'FT wins NF':<12} {'FT wins I1':<12}")
            print("-" * 110)

            for cat in ["newly_solved", "broken", "consistently_correct", "consistently_wrong"]:
                r = results[cat]
                if r["count"] == 0:
                    print(f"{cat:<22} {0:<7}")
                    continue

                ft = r.get("finetune", {}).get("mean", 0)
                nf = r.get("noformat", {}).get("mean", 0)
                i1 = r.get("iter1", {}).get("mean", 0)

                ft_nf_ratio = ft / nf if nf > 0 else 0
                ft_i1_ratio = ft / i1 if i1 > 0 else 0

                ft_wins_nf = r.get("ft_wins_vs_noformat", "N/A")
                ft_wins_i1 = r.get("ft_wins_vs_iter1", "N/A")

                count = r["count"]

                print(f"{cat:<22} {count:<7} {ft:<10.4f} {nf:<10.4f} {i1:<10.4f} {ft_nf_ratio:<8.2f}x {ft_i1_ratio:<8.2f}x {ft_wins_nf}/{count:<10} {ft_wins_i1}/{count}")

    # Cross-model comparison table (baseline vs checkpoints for same categories)
    print("\n\n" + "=" * 80)
    print("INFLUENCE CHANGES BY CATEGORY (Baseline -> Ckpt)")
    print("=" * 80)

    for comparison in ["vs_ckpt32", "vs_ckpt64"]:
        ckpt_name = comparison.replace("vs_", "")
        print(f"\n\n{'#' * 60}")
        print(f"# Changes: Baseline -> {ckpt_name.upper()}")
        print(f"{'#' * 60}")

        if "baseline" not in all_results or comparison not in all_results["baseline"]:
            continue
        if ckpt_name not in all_results or comparison not in all_results[ckpt_name]:
            continue

        bl_results = all_results["baseline"][comparison]
        ckpt_results = all_results[ckpt_name][comparison]

        print(f"\n{'Category':<22} {'BL FT':<9} {'Ck FT':<9} {'FT Chg':<10} {'BL I1':<9} {'Ck I1':<9} {'I1 Chg':<10}")
        print("-" * 85)

        for cat in ["newly_solved", "broken", "consistently_correct", "consistently_wrong"]:
            bl = bl_results[cat]
            ck = ckpt_results[cat]

            if bl["count"] == 0:
                print(f"{cat:<22} (no samples)")
                continue

            bl_ft = bl.get("finetune", {}).get("mean", 0)
            ck_ft = ck.get("finetune", {}).get("mean", 0)
            ft_chg = ((ck_ft - bl_ft) / bl_ft * 100) if bl_ft > 0 else 0

            bl_i1 = bl.get("iter1", {}).get("mean", 0)
            ck_i1 = ck.get("iter1", {}).get("mean", 0)
            i1_chg = ((ck_i1 - bl_i1) / bl_i1 * 100) if bl_i1 > 0 else 0

            print(f"{cat:<22} {bl_ft:<9.4f} {ck_ft:<9.4f} {ft_chg:>+8.1f}% {bl_i1:<9.4f} {ck_i1:<9.4f} {i1_chg:>+8.1f}%")

    # Save results
    output_file = OUTPUT_DIR / "correctness_category_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
