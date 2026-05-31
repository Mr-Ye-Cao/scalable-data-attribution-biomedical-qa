#!/usr/bin/env python3
"""
Compute influence scores for all model-dataset combinations.
Compares pretrain influence vs finetune influence.
"""

import json
import torch
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("/lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune")

# Test and finetune gradients (computed with correct script)
TEST_GRADS_DIR = BASE_DIR / "results/rapidin_aligned_500/grads"
FT_GRADS_DIR = BASE_DIR / "results/rapidin_aligned_500/grads"

# Pretrain gradients from comprehensive run
PT_GRADS_DIR = BASE_DIR / "results/validation/grads"

# Also include iter1 from earlier validation run
ITER1_GRADS_DIR = BASE_DIR / "results/validation/grads/baseline/pretrain_iter1"

OUTPUT_DIR = BASE_DIR / "results/validation/analysis"

MODELS = ["baseline", "ckpt32", "ckpt64"]
DATASETS = ["noformat", "iter1", "iter2", "iter3", "iter4"]


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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("COMPREHENSIVE INFLUENCE ANALYSIS")
    print(f"Time: {datetime.now()}")
    print("=" * 80)

    all_results = {}

    for model in MODELS:
        print(f"\n{'#' * 80}")
        print(f"# MODEL: {model.upper()}")
        print(f"{'#' * 80}")

        # Load test and finetune gradients
        test_grads = load_gradients(TEST_GRADS_DIR / model / "test")
        ft_grads = load_gradients(FT_GRADS_DIR / model / "finetune")

        if test_grads is None or ft_grads is None:
            print(f"Missing test or finetune gradients for {model}")
            continue

        # Compute finetune influence
        ft_infl = compute_influence(test_grads, ft_grads)
        ft_max = ft_infl.max(dim=1).values
        ft_mean = float(ft_max.mean())

        print(f"\nFinetune influence: {ft_mean:.4f}")

        model_results = {
            "finetune": {
                "mean": ft_mean,
                "std": float(ft_max.std()),
                "min": float(ft_max.min()),
                "max": float(ft_max.max()),
            }
        }

        # Load pretrain gradients for each dataset
        for dataset in DATASETS:
            # Special handling for baseline iter1 (different path)
            if model == "baseline" and dataset == "iter1":
                pt_grad_path = ITER1_GRADS_DIR
            else:
                pt_grad_path = PT_GRADS_DIR / model / dataset

            pt_grads = load_gradients(pt_grad_path)

            if pt_grads is None:
                print(f"  {dataset}: Missing gradients at {pt_grad_path}")
                continue

            # Compute influence
            pt_infl = compute_influence(test_grads, pt_grads)
            pt_max = pt_infl.max(dim=1).values
            pt_mean = float(pt_max.mean())

            # FT wins
            ft_wins = (ft_max > pt_max).sum().item()

            ratio = ft_mean / pt_mean if pt_mean > 0 else float('inf')

            print(f"  {dataset}: {pt_mean:.4f} (ratio: {ratio:.2f}x, FT wins: {ft_wins}/500)")

            model_results[dataset] = {
                "mean": pt_mean,
                "std": float(pt_max.std()),
                "min": float(pt_max.min()),
                "max": float(pt_max.max()),
                "ft_pt_ratio": ratio,
                "ft_wins": ft_wins,
            }

        all_results[model] = model_results

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    print(f"\n{'Model':<10} {'Finetune':<12} {'NoFormat':<12} {'Iter1':<12} {'Iter2':<12} {'Iter3':<12} {'Iter4':<12}")
    print(f"{'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    for model in MODELS:
        if model not in all_results:
            continue
        r = all_results[model]
        row = f"{model:<10} "
        row += f"{r.get('finetune', {}).get('mean', 0):<12.4f} "
        for ds in DATASETS:
            val = r.get(ds, {}).get('mean', 0)
            row += f"{val:<12.4f} " if val else f"{'N/A':<12} "
        print(row)

    # Print ratio table
    print(f"\n{'Model':<10} {'NoFormat':<12} {'Iter1':<12} {'Iter2':<12} {'Iter3':<12} {'Iter4':<12}")
    print(f"{'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    for model in MODELS:
        if model not in all_results:
            continue
        r = all_results[model]
        row = f"{model:<10} "
        for ds in DATASETS:
            ratio = r.get(ds, {}).get('ft_pt_ratio', 0)
            row += f"{ratio:<12.2f}x" if ratio else f"{'N/A':<12} "
        print(row)

    # Save results
    output_file = OUTPUT_DIR / "comprehensive_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to: {output_file}")

    # Also save a summary
    summary_file = OUTPUT_DIR / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write("COMPREHENSIVE INFLUENCE ANALYSIS SUMMARY\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("=" * 80 + "\n\n")

        f.write("Datasets:\n")
        f.write("  - noformat: Raw pretrain text without QA template\n")
        f.write("  - iter1: With QA template, all 'maybe' labels\n")
        f.write("  - iter2: Label-matched (yes/no/maybe from source query)\n")
        f.write("  - iter3: Full match (entire output from finetune)\n")
        f.write("  - iter4: No linebreaks in context\n\n")

        f.write("Models:\n")
        f.write("  - baseline: OLMo-3-7B-Instruct (before fine-tuning)\n")
        f.write("  - ckpt32: After 1 epoch fine-tuning\n")
        f.write("  - ckpt64: After 2 epochs fine-tuning (best)\n\n")

        f.write("Results (Mean Max Influence):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<10} {'FT':<10} {'NoFmt':<10} {'Iter1':<10} {'Iter2':<10} {'Iter3':<10} {'Iter4':<10}\n")
        f.write("-" * 80 + "\n")
        for model in MODELS:
            if model not in all_results:
                continue
            r = all_results[model]
            f.write(f"{model:<10} ")
            f.write(f"{r.get('finetune', {}).get('mean', 0):<10.4f} ")
            for ds in DATASETS:
                val = r.get(ds, {}).get('mean', 0)
                f.write(f"{val:<10.4f} " if val else f"{'N/A':<10} ")
            f.write("\n")

    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
