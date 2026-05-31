#!/usr/bin/env python3
"""
Compute influence scores for entertainment (control) data.
Compare with medical pretrain data to validate RapidIn.

Expected result: Medical pretrain should have HIGHER influence than
entertainment data on medical QA tasks, validating that RapidIn
correctly identifies relevant training data.

Now supports multiple checkpoints: baseline, ckpt32 (epoch 1), ckpt64 (epoch 2)
"""

import json
import torch
from pathlib import Path
from datetime import datetime
import argparse

BASE_DIR = Path("/lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune")

# Checkpoint configurations
CHECKPOINTS = {
    "baseline": {
        "test_grads": BASE_DIR / "results/rapidin_aligned_500/grads/baseline/test",
        "medical_iter1": BASE_DIR / "results/validation/grads/baseline/pretrain_iter1",
        "medical_noformat": BASE_DIR / "results/validation/grads/baseline/noformat",
        "entertainment": BASE_DIR / "results/validation/grads/baseline/entertainment",
        "entertainment_noformat": BASE_DIR / "results/validation/grads/baseline/entertainment_noformat",
    },
    "ckpt32": {
        "test_grads": BASE_DIR / "results/rapidin_aligned_500/grads/ckpt32/test",
        "medical_iter1": BASE_DIR / "results/validation/grads/ckpt32/iter1",
        "medical_noformat": BASE_DIR / "results/validation/grads/ckpt32/noformat",
        "entertainment": BASE_DIR / "results/validation/grads/ckpt32/entertainment",
        "entertainment_noformat": BASE_DIR / "results/validation/grads/ckpt32/entertainment_noformat",
    },
    "ckpt64": {
        "test_grads": BASE_DIR / "results/rapidin_aligned_500/grads/ckpt64/test",
        "medical_iter1": BASE_DIR / "results/validation/grads/ckpt64/iter1",
        "medical_noformat": BASE_DIR / "results/validation/grads/ckpt64/noformat",
        "entertainment": BASE_DIR / "results/validation/grads/ckpt64/entertainment",
        "entertainment_noformat": BASE_DIR / "results/validation/grads/ckpt64/entertainment_noformat",
    },
    "ckpt96": {
        "test_grads": BASE_DIR / "results/rapidin_aligned_500/grads/ckpt96/test",
        "medical_iter1": BASE_DIR / "results/validation/grads/ckpt96/iter1",
        "medical_noformat": BASE_DIR / "results/validation/grads/ckpt96/noformat",
        "entertainment": BASE_DIR / "results/validation/grads/ckpt96/entertainment",
        "entertainment_noformat": BASE_DIR / "results/validation/grads/ckpt96/entertainment_noformat",
    },
    "ckpt128": {
        "test_grads": BASE_DIR / "results/rapidin_aligned_500/grads/ckpt128/test",
        "medical_iter1": BASE_DIR / "results/validation/grads/ckpt128/iter1",
        "medical_noformat": BASE_DIR / "results/validation/grads/ckpt128/noformat",
        "entertainment": BASE_DIR / "results/validation/grads/ckpt128/entertainment",
        "entertainment_noformat": BASE_DIR / "results/validation/grads/ckpt128/entertainment_noformat",
    },
    "ckpt160": {
        "test_grads": BASE_DIR / "results/rapidin_aligned_500/grads/ckpt160/test",
        "medical_iter1": BASE_DIR / "results/validation/grads/ckpt160/iter1",
        "medical_noformat": BASE_DIR / "results/validation/grads/ckpt160/noformat",
        "entertainment": BASE_DIR / "results/validation/grads/ckpt160/entertainment",
        "entertainment_noformat": BASE_DIR / "results/validation/grads/ckpt160/entertainment_noformat",
    }
}


def load_gradient(path):
    data = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(data, dict):
        return data.get('grad', data.get('gradient', list(data.values())[0]))
    return data


def load_gradients(grad_dir, max_samples=500):
    grad_dir = Path(grad_dir)
    if not grad_dir.exists():
        print(f"  Warning: Directory not found: {grad_dir}")
        return None
    files = sorted(grad_dir.glob("*.pt"))[:max_samples]
    if len(files) == 0:
        print(f"  Warning: No .pt files found in {grad_dir}")
        return None
    print(f"  Loading {len(files)} gradients from {grad_dir.name}...")
    gradients = []
    for f in files:
        grad = load_gradient(f)
        if isinstance(grad, torch.Tensor):
            gradients.append(grad.float())
        else:
            gradients.append(torch.tensor(grad).float())
    return torch.stack(gradients)


def compute_influence(test_grads, train_grads):
    """Compute cosine similarity between test and train gradients."""
    test_norm = test_grads / (test_grads.norm(dim=1, keepdim=True) + 1e-8)
    train_norm = train_grads / (train_grads.norm(dim=1, keepdim=True) + 1e-8)
    return test_norm @ train_norm.T


def compute_statistics(influence_matrix):
    """Compute statistics from influence matrix."""
    max_per_query = influence_matrix.max(dim=1).values
    return {
        "mean_max": float(max_per_query.mean()),
        "std_max": float(max_per_query.std()),
        "min_max": float(max_per_query.min()),
        "max_max": float(max_per_query.max()),
        "median_max": float(max_per_query.median()),
        "max_per_query": max_per_query.tolist()
    }


def process_checkpoint(ckpt_name, paths, output_dir):
    """Process a single checkpoint and compute influence scores."""
    print(f"\n{'=' * 80}")
    print(f"PROCESSING: {ckpt_name.upper()}")
    print("=" * 80)

    # Load test gradients
    print("\nLoading test gradients...")
    test_grads = load_gradients(paths["test_grads"])
    if test_grads is None:
        print(f"ERROR: Cannot load test gradients for {ckpt_name}!")
        return None
    print(f"  Test gradients shape: {test_grads.shape}")

    results = {}
    datasets = [
        ("entertainment", paths.get("entertainment")),
        ("entertainment_noformat", paths.get("entertainment_noformat")),
        ("medical_iter1", paths.get("medical_iter1")),
        ("medical_noformat", paths.get("medical_noformat")),
    ]

    for name, grad_path in datasets:
        if grad_path is None:
            continue
        print(f"\n{'-' * 40}")
        print(f"{name.upper()}:")
        grads = load_gradients(grad_path)
        if grads is not None:
            print(f"  Shape: {grads.shape}")
            infl = compute_influence(test_grads, grads)
            stats = compute_statistics(infl)
            results[name] = stats
            print(f"  Mean Max Influence: {stats['mean_max']:.4f}")
            print(f"  Std: {stats['std_max']:.4f}")
        else:
            print(f"  Warning: Cannot load {name} gradients")

    # Save results
    ckpt_output_dir = output_dir / ckpt_name
    ckpt_output_dir.mkdir(parents=True, exist_ok=True)

    for key in ["entertainment", "entertainment_noformat"]:
        if key in results:
            output_file = ckpt_output_dir / f"{key}.json"
            output_data = {
                "model": ckpt_name,
                "dataset": key,
                "n_test": 500,
                "n_train": len(results[key].get("max_per_query", [])),
                "summary": {
                    "mean_max": results[key]["mean_max"],
                    "std_max": results[key]["std_max"],
                    "min_max": results[key]["min_max"],
                    "max_max": results[key]["max_max"],
                    "median_max": results[key]["median_max"]
                },
                "per_query": [
                    {"test_idx": i, "max_influence": score}
                    for i, score in enumerate(results[key]["max_per_query"])
                ]
            }
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nSaved: {output_file}")

    return results


def print_comparison(all_results):
    """Print comparison summary across all checkpoints."""
    print("\n" + "=" * 100)
    print("CROSS-CHECKPOINT COMPARISON (Entertainment vs Medical × Formatted vs NoFormat)")
    print("=" * 100)

    # Build comparison table
    print(f"\n{'Model':<12} {'Ent+Fmt':<12} {'Ent NoFmt':<12} {'Med+Fmt':<12} {'Med NoFmt':<12} {'Fmt Effect':<12}")
    print("-" * 72)

    comparison_data = {}
    for ckpt in ["baseline", "ckpt32", "ckpt64", "ckpt96", "ckpt128", "ckpt160"]:
        if ckpt not in all_results:
            continue
        res = all_results[ckpt]
        ent_fmt = res.get("entertainment", {}).get("mean_max", 0)
        ent_nf = res.get("entertainment_noformat", {}).get("mean_max", 0)
        med_fmt = res.get("medical_iter1", {}).get("mean_max", 0)
        med_nf = res.get("medical_noformat", {}).get("mean_max", 0)

        fmt_effect = ent_fmt / ent_nf if ent_nf > 0 else 0

        print(f"{ckpt:<12} {ent_fmt:<12.4f} {ent_nf:<12.4f} {med_fmt:<12.4f} {med_nf:<12.4f} {fmt_effect:<12.2f}x")

        comparison_data[ckpt] = {
            "entertainment_formatted": ent_fmt,
            "entertainment_noformat": ent_nf,
            "medical_formatted": med_fmt,
            "medical_noformat": med_nf,
            "format_effect": fmt_effect
        }

    # Show trends
    print("\n" + "-" * 72)
    print("TRAINING DYNAMICS:")

    if "baseline" in comparison_data and "ckpt160" in comparison_data:
        base = comparison_data["baseline"]
        final = comparison_data["ckpt160"]

        for data_type in ["entertainment_formatted", "entertainment_noformat", "medical_formatted", "medical_noformat"]:
            if base[data_type] > 0:
                change = ((final[data_type] - base[data_type]) / base[data_type]) * 100
                direction = "↑" if change > 0 else "↓"
                print(f"  {data_type:<25}: {base[data_type]:.4f} → {final[data_type]:.4f} ({direction}{abs(change):.1f}%)")

    return comparison_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", choices=["baseline", "ckpt32", "ckpt64", "ckpt96", "ckpt128", "ckpt160", "all"],
                        default="all", help="Which checkpoint to process")
    args = parser.parse_args()

    print("=" * 80)
    print("ENTERTAINMENT vs MEDICAL PRETRAIN INFLUENCE COMPARISON")
    print("=" * 80)
    print(f"\nTime: {datetime.now()}")
    print(f"\nPurpose: Validate that RapidIn correctly identifies")
    print("         medical data as more influential for medical QA tasks")
    print("\nExpected: Medical pretrain > Entertainment data")

    output_dir = BASE_DIR / "data-aggregate/outputs/influence_scores"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    checkpoints_to_process = [args.checkpoint] if args.checkpoint != "all" else ["baseline", "ckpt32", "ckpt64", "ckpt96", "ckpt128", "ckpt160"]

    for ckpt in checkpoints_to_process:
        if ckpt in CHECKPOINTS:
            results = process_checkpoint(ckpt, CHECKPOINTS[ckpt], output_dir)
            if results:
                all_results[ckpt] = results

    # Print cross-checkpoint comparison if we have multiple
    if len(all_results) > 1:
        comparison_data = print_comparison(all_results)

        # Save comparison
        comparison_file = BASE_DIR / "data-aggregate/outputs/analysis" / "entertainment_comparison_all_checkpoints.json"
        comparison_file.parent.mkdir(parents=True, exist_ok=True)
        with open(comparison_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "checkpoints": comparison_data
            }, f, indent=2)
        print(f"\nComparison saved to: {comparison_file}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
