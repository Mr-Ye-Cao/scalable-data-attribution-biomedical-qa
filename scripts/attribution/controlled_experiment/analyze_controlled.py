#!/usr/bin/env python3
"""
Step 6: Analyze Controlled Attribution Experiment

Compare influence scores between:
- Pretrain (with QA template) vs Finetune (with QA template)

Both datasets now have identical template structure, allowing fair comparison.

Categories:
1. Newly Solved: Baseline wrong -> Checkpoint correct
2. Broken: Baseline correct -> Checkpoint wrong
3. Consistently Correct: Both correct
4. Consistently Wrong: Both wrong
"""

import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

BASE_DIR = Path("/lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune")

# Paths - controlled experiment uses template-aligned pretrain gradients
PREDICTIONS_DIR = BASE_DIR / "eval_output_results"
ALIGNED_GRADS = BASE_DIR / "results/rapidin_aligned_500/grads"  # finetune & test
CONTROLLED_GRADS = BASE_DIR / "results/controlled_experiment/grads"  # pretrain with template

OUTPUT_DIR = BASE_DIR / "results/controlled_experiment/analysis"


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
    """Load a single gradient file."""
    data = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(data, dict):
        return data.get('grad', data.get('gradient', list(data.values())[0]))
    return data


def load_gradients(grad_dir, max_samples=500):
    """Load gradients from a directory."""
    grad_dir = Path(grad_dir)
    if not grad_dir.exists():
        print(f"WARNING: {grad_dir} not found")
        return None

    files = sorted(grad_dir.glob("*.pt"))[:max_samples]
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
    """Compute normalized influence scores (cosine similarity)."""
    test_norm = test_grads / (test_grads.norm(dim=1, keepdim=True) + 1e-8)
    train_norm = train_grads / (train_grads.norm(dim=1, keepdim=True) + 1e-8)
    return test_norm @ train_norm.T


def categorize_queries(baseline_preds, ckpt_preds, max_queries=500):
    """Categorize queries based on correctness transition."""
    pmids = list(baseline_preds.keys())[:max_queries]

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


def analyze_influence_by_category(categories, finetune_infl, pretrain_infl):
    """Analyze influence statistics for each category."""
    results = {}

    for cat_name, queries in categories.items():
        if len(queries) == 0:
            results[cat_name] = {"count": 0}
            continue

        indices = [q[0] for q in queries]

        # Get influence scores for this category
        ft_scores = finetune_infl[indices]  # [num_queries, num_finetune]
        pt_scores = pretrain_infl[indices]  # [num_queries, num_pretrain]

        # Compute per-query max influence
        ft_max = ft_scores.max(dim=1).values
        pt_max = pt_scores.max(dim=1).values

        # Compute per-query mean influence
        ft_mean = ft_scores.mean(dim=1)
        pt_mean = pt_scores.mean(dim=1)

        # Compare FT vs PT for each query
        ft_dominant = (ft_max > pt_max).sum().item()
        pt_dominant = (pt_max > ft_max).sum().item()

        results[cat_name] = {
            "count": len(queries),
            "finetune": {
                "max_mean": float(ft_max.mean()),
                "max_std": float(ft_max.std()),
                "max_median": float(ft_max.median()),
                "mean_mean": float(ft_mean.mean()),
                "global_max": float(ft_scores.max()),
            },
            "pretrain": {
                "max_mean": float(pt_max.mean()),
                "max_std": float(pt_max.std()),
                "max_median": float(pt_max.median()),
                "mean_mean": float(pt_mean.mean()),
                "global_max": float(pt_scores.max()),
            },
            "ft_dominant": ft_dominant,
            "pt_dominant": pt_dominant,
            "ft_ratio": float(ft_max.mean() / (pt_max.mean() + 1e-8)),
            "pmids": [q[1] for q in queries],
        }

    return results


def print_results(model_name, results):
    """Print formatted results."""
    print(f"\n{'='*80}")
    print(f"CONTROLLED EXPERIMENT: {model_name.upper()} Analysis")
    print(f"(Both pretrain and finetune have identical QA template)")
    print(f"{'='*80}")

    print(f"\n{'Category':<25} {'Count':<7} {'FT Max':<10} {'PT Max':<10} {'FT/PT':<8} {'FT Wins':<12}")
    print(f"{'-'*25} {'-'*7} {'-'*10} {'-'*10} {'-'*8} {'-'*12}")

    for cat in ["newly_solved", "broken", "consistently_correct", "consistently_wrong"]:
        r = results[cat]
        if r["count"] == 0:
            print(f"{cat:<25} {0:<7}")
            continue

        ft_max = r["finetune"]["max_mean"]
        pt_max = r["pretrain"]["max_mean"]
        ratio = r["ft_ratio"]
        ft_wins = r["ft_dominant"]
        count = r["count"]

        print(f"{cat:<25} {count:<7} {ft_max:<10.4f} {pt_max:<10.4f} {ratio:<8.2f}x {ft_wins}/{count} ({100*ft_wins/count:.0f}%)")


def main():
    print("=" * 80)
    print("CONTROLLED ATTRIBUTION EXPERIMENT - Step 6 Analysis")
    print("=" * 80)
    print("\nThis experiment compares pretrain vs finetune influence")
    print("with IDENTICAL template structure (QA format) on both datasets.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load predictions
    print("\n" + "-" * 40)
    print("Loading predictions...")
    baseline_preds = load_predictions("baseline")
    ckpt32_preds = load_predictions("ckpt32")
    ckpt64_preds = load_predictions("ckpt64")

    all_results = {}

    for model_name, ckpt_preds in [("baseline", baseline_preds), ("ckpt32", ckpt32_preds), ("ckpt64", ckpt64_preds)]:
        print(f"\n{'#'*80}")
        print(f"# Processing {model_name.upper()}")
        print(f"{'#'*80}")

        # Load gradients
        print(f"\nLoading gradients for {model_name}...")

        # Test & finetune from aligned_500 (standard location)
        test_grads = load_gradients(ALIGNED_GRADS / model_name / "test")
        finetune_grads = load_gradients(ALIGNED_GRADS / model_name / "finetune")

        # Pretrain from controlled experiment (with template)
        pretrain_grads = load_gradients(CONTROLLED_GRADS / model_name / "pretrain")

        if test_grads is None or finetune_grads is None or pretrain_grads is None:
            print(f"Skipping {model_name} - missing gradients")
            continue

        print(f"  Test gradients: {test_grads.shape}")
        print(f"  Finetune gradients: {finetune_grads.shape}")
        print(f"  Pretrain gradients: {pretrain_grads.shape}")

        # Compute influence scores
        print("\nComputing influence scores...")
        finetune_infl = compute_influence(test_grads, finetune_grads)
        pretrain_infl = compute_influence(test_grads, pretrain_grads)

        print(f"  Finetune influence shape: {finetune_infl.shape}")
        print(f"  Pretrain influence shape: {pretrain_infl.shape}")

        # For baseline, compare to itself (categories won't be meaningful)
        if model_name == "baseline":
            # Just compute overall statistics for baseline
            print("\nBaseline influence statistics:")
            print(f"  Finetune max: {finetune_infl.max():.4f}, mean: {finetune_infl.mean():.4f}")
            print(f"  Pretrain max: {pretrain_infl.max():.4f}, mean: {pretrain_infl.mean():.4f}")

            # Per-query analysis
            ft_max_per_q = finetune_infl.max(dim=1).values
            pt_max_per_q = pretrain_infl.max(dim=1).values
            ft_dominant = (ft_max_per_q > pt_max_per_q).sum().item()

            all_results["baseline"] = {
                "finetune_mean": float(finetune_infl.mean()),
                "pretrain_mean": float(pretrain_infl.mean()),
                "finetune_max_per_query_mean": float(ft_max_per_q.mean()),
                "pretrain_max_per_query_mean": float(pt_max_per_q.mean()),
                "ft_dominant": ft_dominant,
                "total_queries": 500,
            }
            continue

        # Categorize queries (comparing baseline to this checkpoint)
        categories = categorize_queries(baseline_preds, ckpt_preds)

        print(f"\nCategory distribution:")
        for cat, queries in categories.items():
            print(f"  {cat}: {len(queries)}")

        # Analyze by category
        results = analyze_influence_by_category(categories, finetune_infl, pretrain_infl)

        # Print results
        print_results(model_name, results)

        all_results[model_name] = {
            "categories": {
                cat: {k: v for k, v in r.items() if k != "pmids"}
                for cat, r in results.items()
            },
            "category_counts": {cat: len(queries) for cat, queries in categories.items()},
        }

    # Cross-model comparison
    print(f"\n\n{'#'*80}")
    print("# CROSS-MODEL COMPARISON")
    print(f"{'#'*80}")

    print(f"\n{'Category':<25} {'Model':<10} {'FT Max':<10} {'PT Max':<10} {'FT/PT':<8} {'FT Wins'}")
    print(f"{'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*12}")

    for cat in ["newly_solved", "broken", "consistently_correct", "consistently_wrong"]:
        for model_name in ["ckpt32", "ckpt64"]:
            if model_name not in all_results:
                continue
            r = all_results[model_name]["categories"].get(cat, {"count": 0})
            if r.get("count", 0) == 0:
                continue

            ft_max = r["finetune"]["max_mean"]
            pt_max = r["pretrain"]["max_mean"]
            ratio = r["ft_ratio"]
            ft_wins = r["ft_dominant"]
            count = r["count"]

            cat_label = cat if model_name == "ckpt32" else ""
            print(f"{cat_label:<25} {model_name:<10} {ft_max:<10.4f} {pt_max:<10.4f} {ratio:<8.2f}x {ft_wins}/{count}")

    # Key findings
    print(f"\n\n{'#'*80}")
    print("# KEY FINDINGS")
    print(f"{'#'*80}")

    for model_name in ["ckpt32", "ckpt64"]:
        if model_name not in all_results:
            continue

        print(f"\n## {model_name.upper()}")

        cats = all_results[model_name]["categories"]

        # Newly solved analysis
        if cats.get("newly_solved", {}).get("count", 0) > 0:
            ns = cats["newly_solved"]
            print(f"  NEWLY SOLVED ({ns['count']} queries):")
            print(f"    - Finetune influence: {ns['finetune']['max_mean']:.4f}")
            print(f"    - Pretrain influence: {ns['pretrain']['max_mean']:.4f}")
            print(f"    - FT/PT ratio: {ns['ft_ratio']:.2f}x")
            print(f"    - FT dominant in {ns['ft_dominant']}/{ns['count']} queries ({100*ns['ft_dominant']/ns['count']:.0f}%)")

        # Broken analysis
        if cats.get("broken", {}).get("count", 0) > 0:
            br = cats["broken"]
            print(f"  BROKEN ({br['count']} queries):")
            print(f"    - Finetune influence: {br['finetune']['max_mean']:.4f}")
            print(f"    - Pretrain influence: {br['pretrain']['max_mean']:.4f}")
            print(f"    - FT/PT ratio: {br['ft_ratio']:.2f}x")

    # Save results
    output_file = OUTPUT_DIR / "controlled_analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to: {output_file}")

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
