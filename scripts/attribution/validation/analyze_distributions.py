#!/usr/bin/env python3
"""
Analyze influence score distributions across models and datasets.
Compares noformat vs iter1 vs finetune influence.
Generates visualizations and statistics for OBSERVATIONS_CORRECTED.md.
"""

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

BASE_DIR = Path("/lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune")

# Gradients paths
TEST_GRADS_DIR = BASE_DIR / "results/rapidin_aligned_500/grads"
FT_GRADS_DIR = BASE_DIR / "results/rapidin_aligned_500/grads"
PT_GRADS_DIR = BASE_DIR / "results/validation/grads"

OUTPUT_DIR = BASE_DIR / "results/validation/analysis"
FIGURES_DIR = OUTPUT_DIR / "figures"

MODELS = ["baseline", "ckpt32", "ckpt64"]
MODEL_LABELS = {
    "baseline": "Baseline (41.8%)",
    "ckpt32": "Epoch 1 (69.0%)",
    "ckpt64": "Epoch 2 (74.4%)"
}


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
    """Compute cosine similarity based influence scores."""
    test_norm = test_grads / (test_grads.norm(dim=1, keepdim=True) + 1e-8)
    train_norm = train_grads / (train_grads.norm(dim=1, keepdim=True) + 1e-8)
    return test_norm @ train_norm.T


def compute_stats(values):
    """Compute comprehensive statistics for a set of values."""
    values = np.array(values)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("INFLUENCE DISTRIBUTION ANALYSIS")
    print(f"Time: {datetime.now()}")
    print("=" * 80)

    # Store all influence scores for analysis
    all_data = {}

    for model in MODELS:
        print(f"\n{'#' * 60}")
        print(f"# MODEL: {model.upper()}")
        print(f"{'#' * 60}")

        # Load test gradients
        test_grads = load_gradients(TEST_GRADS_DIR / model / "test")
        if test_grads is None:
            print(f"Missing test gradients for {model}")
            continue

        all_data[model] = {}

        # Load and compute finetune influence
        ft_grads = load_gradients(FT_GRADS_DIR / model / "finetune")
        if ft_grads is not None:
            ft_infl = compute_influence(test_grads, ft_grads)
            ft_max = ft_infl.max(dim=1).values.numpy()
            ft_mean_per_test = ft_infl.mean(dim=1).numpy()
            all_data[model]["finetune"] = {
                "max_per_test": ft_max,
                "mean_per_test": ft_mean_per_test,
                "all_scores": ft_infl.numpy().flatten(),
                "stats": compute_stats(ft_max)
            }
            print(f"\nFinetune: mean_max={np.mean(ft_max):.4f}, std={np.std(ft_max):.4f}")

        # Load and compute noformat influence
        nf_grads = load_gradients(PT_GRADS_DIR / model / "noformat")
        if nf_grads is not None:
            nf_infl = compute_influence(test_grads, nf_grads)
            nf_max = nf_infl.max(dim=1).values.numpy()
            nf_mean_per_test = nf_infl.mean(dim=1).numpy()
            all_data[model]["noformat"] = {
                "max_per_test": nf_max,
                "mean_per_test": nf_mean_per_test,
                "all_scores": nf_infl.numpy().flatten(),
                "stats": compute_stats(nf_max)
            }
            print(f"NoFormat: mean_max={np.mean(nf_max):.4f}, std={np.std(nf_max):.4f}")

        # Load and compute iter1 influence (check both possible paths)
        iter1_grads = load_gradients(PT_GRADS_DIR / model / "iter1")
        if iter1_grads is None:
            iter1_grads = load_gradients(PT_GRADS_DIR / model / "pretrain_iter1")
        if iter1_grads is not None:
            iter1_infl = compute_influence(test_grads, iter1_grads)
            iter1_max = iter1_infl.max(dim=1).values.numpy()
            iter1_mean_per_test = iter1_infl.mean(dim=1).numpy()
            all_data[model]["iter1"] = {
                "max_per_test": iter1_max,
                "mean_per_test": iter1_mean_per_test,
                "all_scores": iter1_infl.numpy().flatten(),
                "stats": compute_stats(iter1_max)
            }
            print(f"Iter1:    mean_max={np.mean(iter1_max):.4f}, std={np.std(iter1_max):.4f}")

    # =========================================================================
    # Generate Visualizations
    # =========================================================================
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    # Figure 1: Box plots comparing distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, model in enumerate(MODELS):
        if model not in all_data:
            continue
        ax = axes[idx]
        data_to_plot = []
        labels = []

        for dataset in ["finetune", "noformat", "iter1"]:
            if dataset in all_data[model]:
                data_to_plot.append(all_data[model][dataset]["max_per_test"])
                labels.append(dataset.capitalize() if dataset != "noformat" else "NoFormat")

        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(MODEL_LABELS[model])
        ax.set_ylabel("Max Influence Score")
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle("Influence Score Distributions (Max per Test Query)", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "boxplot_distributions.png", dpi=150)
    plt.close()
    print("Saved: boxplot_distributions.png")

    # Figure 2: Histogram comparison
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    datasets = ["finetune", "noformat", "iter1"]

    for row, model in enumerate(MODELS):
        if model not in all_data:
            continue
        for col, dataset in enumerate(datasets):
            ax = axes[row, col]
            if dataset in all_data[model]:
                scores = all_data[model][dataset]["max_per_test"]
                ax.hist(scores, bins=30, alpha=0.7, color=['#2ecc71', '#e74c3c', '#3498db'][col])
                ax.axvline(np.mean(scores), color='black', linestyle='--', label=f'Mean: {np.mean(scores):.3f}')
                ax.set_xlabel("Max Influence Score")
                ax.set_ylabel("Count")
                ax.legend(fontsize=8)
            ax.set_title(f"{MODEL_LABELS[model]}\n{dataset.capitalize()}")

    plt.suptitle("Influence Score Histograms", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "histogram_distributions.png", dpi=150)
    plt.close()
    print("Saved: histogram_distributions.png")

    # Figure 3: Trend across models
    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(MODELS))
    width = 0.25

    ft_means = [all_data[m]["finetune"]["stats"]["mean"] if "finetune" in all_data[m] else 0 for m in MODELS if m in all_data]
    nf_means = [all_data[m]["noformat"]["stats"]["mean"] if "noformat" in all_data[m] else 0 for m in MODELS if m in all_data]
    iter1_means = [all_data[m]["iter1"]["stats"]["mean"] if "iter1" in all_data[m] else 0 for m in MODELS if m in all_data]

    ft_stds = [all_data[m]["finetune"]["stats"]["std"] if "finetune" in all_data[m] else 0 for m in MODELS if m in all_data]
    nf_stds = [all_data[m]["noformat"]["stats"]["std"] if "noformat" in all_data[m] else 0 for m in MODELS if m in all_data]
    iter1_stds = [all_data[m]["iter1"]["stats"]["std"] if "iter1" in all_data[m] else 0 for m in MODELS if m in all_data]

    ax.bar(x_pos - width, ft_means, width, yerr=ft_stds, label='Finetune', color='#2ecc71', capsize=5)
    ax.bar(x_pos, nf_means, width, yerr=nf_stds, label='NoFormat', color='#e74c3c', capsize=5)
    ax.bar(x_pos + width, iter1_means, width, yerr=iter1_stds, label='Iter1', color='#3498db', capsize=5)

    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Max Influence Score")
    ax.set_title("Influence Score Trends Across Training")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "trend_across_models.png", dpi=150)
    plt.close()
    print("Saved: trend_across_models.png")

    # Figure 4: FT/PT Ratio trend
    fig, ax = plt.subplots(figsize=(10, 6))

    nf_ratios = []
    iter1_ratios = []
    for m in MODELS:
        if m not in all_data:
            continue
        if "finetune" in all_data[m] and "noformat" in all_data[m]:
            nf_ratios.append(all_data[m]["finetune"]["stats"]["mean"] / all_data[m]["noformat"]["stats"]["mean"])
        else:
            nf_ratios.append(0)
        if "finetune" in all_data[m] and "iter1" in all_data[m]:
            iter1_ratios.append(all_data[m]["finetune"]["stats"]["mean"] / all_data[m]["iter1"]["stats"]["mean"])
        else:
            iter1_ratios.append(0)

    ax.plot(x_pos, nf_ratios, 'o-', markersize=10, linewidth=2, label='FT/NoFormat Ratio', color='#e74c3c')
    ax.plot(x_pos, iter1_ratios, 's-', markersize=10, linewidth=2, label='FT/Iter1 Ratio', color='#3498db')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Equal influence')

    ax.set_xlabel("Model")
    ax.set_ylabel("FT/PT Ratio")
    ax.set_title("Finetune vs Pretrain Influence Ratio")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
    ax.legend()
    ax.grid(alpha=0.3)

    for i, (nf_r, iter1_r) in enumerate(zip(nf_ratios, iter1_ratios)):
        ax.annotate(f'{nf_r:.2f}x', (i, nf_r), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        ax.annotate(f'{iter1_r:.2f}x', (i, iter1_r), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ratio_trend.png", dpi=150)
    plt.close()
    print("Saved: ratio_trend.png")

    # Figure 5: Scatter plot - FT vs PT per query
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, model in enumerate(MODELS):
        if model not in all_data:
            continue
        ax = axes[idx]

        ft_max = all_data[model]["finetune"]["max_per_test"]

        if "iter1" in all_data[model]:
            iter1_max = all_data[model]["iter1"]["max_per_test"]
            ax.scatter(ft_max, iter1_max, alpha=0.5, s=20, label='Iter1', color='#3498db')

        if "noformat" in all_data[model]:
            nf_max = all_data[model]["noformat"]["max_per_test"]
            ax.scatter(ft_max, nf_max, alpha=0.5, s=20, label='NoFormat', color='#e74c3c')

        # Diagonal line
        lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, label='Equal')

        ax.set_xlabel("Finetune Max Influence")
        ax.set_ylabel("Pretrain Max Influence")
        ax.set_title(MODEL_LABELS[model])
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("Finetune vs Pretrain Influence (Per Test Query)", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "scatter_ft_vs_pt.png", dpi=150)
    plt.close()
    print("Saved: scatter_ft_vs_pt.png")

    # Figure 6: Violin plots
    fig, ax = plt.subplots(figsize=(12, 6))

    positions = []
    data_for_violin = []
    colors_violin = []
    labels_violin = []

    pos = 0
    for model in MODELS:
        if model not in all_data:
            continue
        for dataset, color in [("finetune", '#2ecc71'), ("noformat", '#e74c3c'), ("iter1", '#3498db')]:
            if dataset in all_data[model]:
                positions.append(pos)
                data_for_violin.append(all_data[model][dataset]["max_per_test"])
                colors_violin.append(color)
                labels_violin.append(f"{model}\n{dataset}")
                pos += 1
        pos += 0.5  # Gap between models

    parts = ax.violinplot(data_for_violin, positions=positions, showmeans=True, showmedians=True)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_violin[i])
        pc.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels_violin, fontsize=8)
    ax.set_ylabel("Max Influence Score")
    ax.set_title("Influence Score Distributions (Violin Plot)")
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "violin_distributions.png", dpi=150)
    plt.close()
    print("Saved: violin_distributions.png")

    # =========================================================================
    # Print Summary Statistics
    # =========================================================================
    print("\n" + "=" * 80)
    print("DETAILED STATISTICS")
    print("=" * 80)

    for model in MODELS:
        if model not in all_data:
            continue
        print(f"\n{MODEL_LABELS[model]}:")
        print("-" * 60)

        for dataset in ["finetune", "noformat", "iter1"]:
            if dataset not in all_data[model]:
                continue
            stats = all_data[model][dataset]["stats"]
            print(f"  {dataset:10s}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                  f"median={stats['median']:.4f}, range=[{stats['min']:.4f}, {stats['max']:.4f}], "
                  f"IQR=[{stats['q25']:.4f}, {stats['q75']:.4f}]")

    # FT wins analysis
    print("\n" + "=" * 80)
    print("FINETUNE WINS ANALYSIS")
    print("=" * 80)

    for model in MODELS:
        if model not in all_data:
            continue
        print(f"\n{MODEL_LABELS[model]}:")

        ft_max = all_data[model]["finetune"]["max_per_test"]

        for dataset in ["noformat", "iter1"]:
            if dataset not in all_data[model]:
                continue
            pt_max = all_data[model][dataset]["max_per_test"]
            ft_wins = np.sum(ft_max > pt_max)
            pt_wins = np.sum(pt_max > ft_max)
            ties = np.sum(ft_max == pt_max)
            print(f"  vs {dataset:10s}: FT wins {ft_wins}/500 ({100*ft_wins/500:.1f}%), "
                  f"PT wins {pt_wins}/500 ({100*pt_wins/500:.1f}%)")

    # =========================================================================
    # Save Results
    # =========================================================================

    # Save statistics to JSON (without numpy arrays)
    stats_results = {}
    for model in all_data:
        stats_results[model] = {}
        for dataset in all_data[model]:
            stats_results[model][dataset] = all_data[model][dataset]["stats"]

    with open(OUTPUT_DIR / "distribution_stats.json", 'w') as f:
        json.dump(stats_results, f, indent=2)
    print(f"\nStats saved to: {OUTPUT_DIR / 'distribution_stats.json'}")

    # Save FT wins analysis
    wins_results = {}
    for model in all_data:
        wins_results[model] = {}
        ft_max = all_data[model]["finetune"]["max_per_test"]
        for dataset in ["noformat", "iter1"]:
            if dataset not in all_data[model]:
                continue
            pt_max = all_data[model][dataset]["max_per_test"]
            ft_wins = int(np.sum(ft_max > pt_max))
            pt_wins = int(np.sum(pt_max > ft_max))
            wins_results[model][dataset] = {
                "ft_wins": ft_wins,
                "pt_wins": pt_wins,
                "total": 500
            }

    with open(OUTPUT_DIR / "ft_wins_analysis.json", 'w') as f:
        json.dump(wins_results, f, indent=2)
    print(f"FT wins analysis saved to: {OUTPUT_DIR / 'ft_wins_analysis.json'}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
