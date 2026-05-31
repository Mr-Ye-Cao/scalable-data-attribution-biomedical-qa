#!/usr/bin/env python3
"""
Visualize influence score patterns across correctness categories.
"""

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path("/lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune")
OUTPUT_DIR = BASE_DIR / "results/figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Paths
PREDICTIONS_DIR = BASE_DIR / "eval_output_results"
ALIGNED_GRADS = BASE_DIR / "results/rapidin_aligned_500/grads"
ORIGINAL_GRADS = BASE_DIR / "results/rapidin_original/grads"

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'newly_solved': '#2ecc71',      # Green
    'broken': '#e74c3c',            # Red
    'consistently_correct': '#3498db',  # Blue
    'consistently_wrong': '#95a5a6',    # Gray
}
CATEGORY_LABELS = {
    'newly_solved': 'Newly Solved',
    'broken': 'Broken',
    'consistently_correct': 'Consistently Correct',
    'consistently_wrong': 'Consistently Wrong',
}


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
        return None
    files = sorted(grad_dir.glob("*.pt"))
    if max_samples:
        files = files[:max_samples]
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
    """Compute normalized influence scores."""
    test_norm = test_grads / (test_grads.norm(dim=1, keepdim=True) + 1e-8)
    train_norm = train_grads / (train_grads.norm(dim=1, keepdim=True) + 1e-8)
    return test_norm @ train_norm.T


def categorize_queries(baseline_preds, ckpt_preds):
    """Categorize queries based on correctness transition."""
    pmids = list(baseline_preds.keys())[:500]
    categories = {
        "newly_solved": [], "broken": [],
        "consistently_correct": [], "consistently_wrong": [],
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


def plot_influence_change_bars(all_data):
    """Plot bar chart of influence changes by category."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    categories = ['newly_solved', 'broken', 'consistently_correct', 'consistently_wrong']
    x = np.arange(len(categories))
    width = 0.35

    for ax_idx, (model_name, model_label) in enumerate([('ckpt32', 'Epoch 1 (Ckpt32)'), ('ckpt64', 'Epoch 2 (Ckpt64)'), ('ckpt96', 'Epoch 3 (Ckpt96)')]):
        ax = axes[ax_idx]

        ft_changes = []
        pt_changes = []

        for cat in categories:
            data = all_data[model_name][cat]
            ft_changes.append(data['ft_change'])
            pt_changes.append(data['pt_change'])

        bars1 = ax.bar(x - width/2, ft_changes, width, label='Finetune', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, pt_changes, width, label='Pretrain', color='#e67e22', alpha=0.8)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Influence Change from Baseline (%)', fontsize=12)
        ax.set_title(f'{model_label}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([CATEGORY_LABELS[c] for c in categories], rotation=15, ha='right')
        ax.legend()
        ax.set_ylim(-45, 25)

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3 if height >= 0 else -12), textcoords="offset points",
                       ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3 if height >= 0 else -12), textcoords="offset points",
                       ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'influence_change_by_category.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'influence_change_by_category.pdf', bbox_inches='tight')
    print(f"Saved: influence_change_by_category.png/pdf")
    plt.close()


def plot_scatter_ft_vs_pt(all_influence_data):
    """Scatter plot of finetune vs pretrain influence per query, colored by category."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    models = ['baseline', 'ckpt32', 'ckpt64', 'ckpt96']
    model_labels = ['Baseline', 'Epoch 1 (Ckpt32)', 'Epoch 2 (Ckpt64)', 'Epoch 3 (Ckpt96)']

    for ax_idx, (model_name, model_label) in enumerate(zip(models, model_labels)):
        ax = axes[ax_idx]

        data = all_influence_data[model_name]

        for cat in ['newly_solved', 'broken', 'consistently_correct', 'consistently_wrong']:
            if cat not in data or len(data[cat]['ft_max']) == 0:
                continue
            ax.scatter(data[cat]['pt_max'], data[cat]['ft_max'],
                      c=COLORS[cat], label=CATEGORY_LABELS[cat], alpha=0.7, s=60, edgecolors='white', linewidth=0.5)

        ax.set_xlabel('Pretrain Influence (max per query)', fontsize=11)
        ax.set_ylabel('Finetune Influence (max per query)', fontsize=11)
        ax.set_title(f'{model_label}', fontsize=13, fontweight='bold')

        # Add diagonal line (FT = PT)
        lims = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'k--', alpha=0.3, label='FT = PT')
        ax.set_xlim(0, 0.2)
        ax.set_ylim(0, 0.6)

        if ax_idx == 3:
            ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'scatter_ft_vs_pt.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'scatter_ft_vs_pt.pdf', bbox_inches='tight')
    print(f"Saved: scatter_ft_vs_pt.png/pdf")
    plt.close()


def plot_influence_evolution(all_influence_data):
    """Line plot showing how influence evolves across training for each category."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    models = ['baseline', 'ckpt32', 'ckpt64', 'ckpt96']
    x_labels = ['Baseline', 'Epoch 1', 'Epoch 2', 'Epoch 3']
    x = np.arange(len(models))

    categories = ['newly_solved', 'broken', 'consistently_correct', 'consistently_wrong']

    # Plot Finetune evolution
    ax = axes[0]
    for cat in categories:
        values = []
        for model in models:
            if cat in all_influence_data[model] and len(all_influence_data[model][cat]['ft_max']) > 0:
                values.append(np.mean(all_influence_data[model][cat]['ft_max']))
            else:
                values.append(np.nan)
        ax.plot(x, values, 'o-', color=COLORS[cat], label=CATEGORY_LABELS[cat], linewidth=2, markersize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel('Finetune Influence (max per query mean)', fontsize=11)
    ax.set_title('Finetune Influence Evolution', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(0.25, 0.5)

    # Plot Pretrain evolution
    ax = axes[1]
    for cat in categories:
        values = []
        for model in models:
            if cat in all_influence_data[model] and len(all_influence_data[model][cat]['pt_max']) > 0:
                values.append(np.mean(all_influence_data[model][cat]['pt_max']))
            else:
                values.append(np.nan)
        ax.plot(x, values, 'o-', color=COLORS[cat], label=CATEGORY_LABELS[cat], linewidth=2, markersize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel('Pretrain Influence (max per query mean)', fontsize=11)
    ax.set_title('Pretrain Influence Evolution', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(0.05, 0.12)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'influence_evolution.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'influence_evolution.pdf', bbox_inches='tight')
    print(f"Saved: influence_evolution.png/pdf")
    plt.close()


def plot_category_distribution(categories_32, categories_64, categories_96):
    """Stacked bar showing category distribution for each checkpoint."""
    fig, ax = plt.subplots(figsize=(12, 6))

    cats = ['newly_solved', 'broken', 'consistently_correct', 'consistently_wrong']

    ckpt32_counts = [len(categories_32[c]) for c in cats]
    ckpt64_counts = [len(categories_64[c]) for c in cats]
    ckpt96_counts = [len(categories_96[c]) for c in cats]

    x = np.arange(3)
    width = 0.6

    bottoms = [0, 0, 0]
    all_counts = [ckpt32_counts, ckpt64_counts, ckpt96_counts]

    for i, cat in enumerate(cats):
        for j in range(3):
            ax.bar(j, all_counts[j][i], width, bottom=bottoms[j], color=COLORS[cat],
                   label=CATEGORY_LABELS[cat] if (i == 0 and j == 0) else None)
            # Add count labels
            if all_counts[j][i] > 2:
                ax.text(j, bottoms[j] + all_counts[j][i]/2, str(all_counts[j][i]),
                       ha='center', va='center', fontsize=11, fontweight='bold', color='white')
            bottoms[j] += all_counts[j][i]

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Epoch 1 (Ckpt32)\n69.0% accuracy', 'Epoch 2 (Ckpt64)\n74.4% accuracy', 'Epoch 3 (Ckpt96)\n69.0% accuracy'], fontsize=12)
    ax.set_ylabel('Number of Test Queries', fontsize=12)
    ax.set_title('Query Categorization: Baseline → Checkpoint', fontsize=14, fontweight='bold')

    # Custom legend
    patches = [mpatches.Patch(color=COLORS[c], label=CATEGORY_LABELS[c]) for c in cats]
    ax.legend(handles=patches, loc='upper right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'category_distribution.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'category_distribution.pdf', bbox_inches='tight')
    print(f"Saved: category_distribution.png/pdf")
    plt.close()


def plot_broken_queries_detail(all_influence_data):
    """Special focus plot on broken queries showing influence changes."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get broken query data
    models = ['baseline', 'ckpt32', 'ckpt64', 'ckpt96']
    x = np.arange(len(models))
    x_labels = ['Baseline', 'Epoch 1', 'Epoch 2', 'Epoch 3']

    # For broken queries
    ft_values = []
    pt_values = []
    for model in models:
        if 'broken' in all_influence_data[model] and len(all_influence_data[model]['broken']['ft_max']) > 0:
            ft_values.append(np.mean(all_influence_data[model]['broken']['ft_max']))
            pt_values.append(np.mean(all_influence_data[model]['broken']['pt_max']))
        else:
            ft_values.append(np.nan)
            pt_values.append(np.nan)

    width = 0.35
    bars1 = ax.bar(x - width/2, ft_values, width, label='Finetune Influence', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, pt_values, width, label='Pretrain Influence', color='#e67e22', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_ylabel('Influence (max per query mean)', fontsize=12)
    ax.set_title('Broken Queries: Influence Pattern Across Training', fontsize=14, fontweight='bold')
    ax.legend()

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if not np.isnan(height):
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        if not np.isnan(height):
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

    # Add annotation showing pretrain change from epoch 1 to epoch 2
    if len(pt_values) > 2 and not np.isnan(pt_values[1]) and not np.isnan(pt_values[2]):
        pt_change_12 = (pt_values[2] - pt_values[1]) / pt_values[1] * 100
        ax.annotate('', xy=(2.175, pt_values[2]), xytext=(1.175, pt_values[1]),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax.text(1.7, (pt_values[1] + pt_values[2])/2 + 0.01, f'{pt_change_12:+.1f}%', fontsize=11, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'broken_queries_detail.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'broken_queries_detail.pdf', bbox_inches='tight')
    print(f"Saved: broken_queries_detail.png/pdf")
    plt.close()


def plot_influence_distribution_violin(all_influence_data):
    """Violin plots showing distribution of influences."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    models = ['baseline', 'ckpt32', 'ckpt64', 'ckpt96']
    model_labels = ['Baseline', 'Epoch 1 (Ckpt32)', 'Epoch 2 (Ckpt64)', 'Epoch 3 (Ckpt96)']
    categories = ['newly_solved', 'broken', 'consistently_correct', 'consistently_wrong']

    for col, (model, model_label) in enumerate(zip(models, model_labels)):
        # Finetune
        ax = axes[0, col]
        data_ft = []
        labels = []
        colors = []
        for cat in categories:
            if cat in all_influence_data[model] and len(all_influence_data[model][cat]['ft_max']) > 0:
                data_ft.append(all_influence_data[model][cat]['ft_max'])
                labels.append(CATEGORY_LABELS[cat].replace(' ', '\n'))
                colors.append(COLORS[cat])

        if data_ft:
            parts = ax.violinplot(data_ft, positions=range(len(data_ft)), showmeans=True, showmedians=True)
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.7)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('Finetune Influence' if col == 0 else '')
        ax.set_title(f'{model_label}', fontsize=12, fontweight='bold')
        ax.set_ylim(0.2, 0.6)

        # Pretrain
        ax = axes[1, col]
        data_pt = []
        labels = []
        colors = []
        for cat in categories:
            if cat in all_influence_data[model] and len(all_influence_data[model][cat]['pt_max']) > 0:
                data_pt.append(all_influence_data[model][cat]['pt_max'])
                labels.append(CATEGORY_LABELS[cat].replace(' ', '\n'))
                colors.append(COLORS[cat])

        if data_pt:
            parts = ax.violinplot(data_pt, positions=range(len(data_pt)), showmeans=True, showmedians=True)
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.7)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('Pretrain Influence' if col == 0 else '')
        ax.set_ylim(0, 0.2)

    plt.suptitle('Influence Distribution by Category', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'influence_distribution_violin.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'influence_distribution_violin.pdf', bbox_inches='tight')
    print(f"Saved: influence_distribution_violin.png/pdf")
    plt.close()


def main():
    print("=" * 70)
    print("Generating Influence Visualizations (4 Models: baseline, ckpt32, ckpt64, ckpt96)")
    print("=" * 70)

    # Load predictions
    print("\nLoading predictions...")
    baseline_preds = load_predictions("baseline")
    ckpt32_preds = load_predictions("ckpt32")
    ckpt64_preds = load_predictions("ckpt64")
    ckpt96_preds = load_predictions("ckpt96")

    # Get categories for each checkpoint
    categories_32 = categorize_queries(baseline_preds, ckpt32_preds)
    categories_64 = categorize_queries(baseline_preds, ckpt64_preds)
    categories_96 = categorize_queries(baseline_preds, ckpt96_preds)

    # Load all gradients
    print("\nLoading gradients...")
    gradients = {}
    for model_name in ['baseline', 'ckpt32', 'ckpt64', 'ckpt96']:
        print(f"  Loading {model_name}...")
        gradients[model_name] = {
            'test': load_gradients(ALIGNED_GRADS / model_name / "test"),
            'finetune': load_gradients(ALIGNED_GRADS / model_name / "finetune", max_samples=500),
            'pretrain': load_gradients(ORIGINAL_GRADS / model_name / "pretrain", max_samples=500),
        }

    # Compute all influence scores
    print("\nComputing influence scores...")
    influence = {}
    for model_name in ['baseline', 'ckpt32', 'ckpt64', 'ckpt96']:
        g = gradients[model_name]
        influence[model_name] = {
            'ft': compute_influence(g['test'], g['finetune']),
            'pt': compute_influence(g['test'], g['pretrain']),
        }

    # Organize data by category for each model (using ckpt96 categories for consistent grouping)
    print("\nOrganizing data by category...")
    all_influence_data = {}

    for model_name in ['baseline', 'ckpt32', 'ckpt64', 'ckpt96']:
        all_influence_data[model_name] = {}
        ft_infl = influence[model_name]['ft']
        pt_infl = influence[model_name]['pt']

        for cat, queries in categories_96.items():
            if len(queries) == 0:
                continue
            indices = [q[0] for q in queries]
            all_influence_data[model_name][cat] = {
                'ft_max': ft_infl[indices].max(dim=1).values.numpy(),
                'pt_max': pt_infl[indices].max(dim=1).values.numpy(),
                'ft_mean': ft_infl[indices].mean(dim=1).numpy(),
                'pt_mean': pt_infl[indices].mean(dim=1).numpy(),
            }

    # Prepare change data for bar charts
    print("\nPreparing change data...")
    all_change_data = {}
    for model_name in ['ckpt32', 'ckpt64', 'ckpt96']:
        all_change_data[model_name] = {}
        for cat in categories_96.keys():
            if cat not in all_influence_data['baseline'] or cat not in all_influence_data[model_name]:
                continue
            bl_ft = np.mean(all_influence_data['baseline'][cat]['ft_max'])
            bl_pt = np.mean(all_influence_data['baseline'][cat]['pt_max'])
            ckpt_ft = np.mean(all_influence_data[model_name][cat]['ft_max'])
            ckpt_pt = np.mean(all_influence_data[model_name][cat]['pt_max'])

            all_change_data[model_name][cat] = {
                'ft_change': (ckpt_ft - bl_ft) / bl_ft * 100 if bl_ft > 0 else 0,
                'pt_change': (ckpt_pt - bl_pt) / bl_pt * 100 if bl_pt > 0 else 0,
            }

    # Generate plots
    print("\nGenerating plots...")

    print("  1. Category distribution...")
    plot_category_distribution(categories_32, categories_64, categories_96)

    print("  2. Influence change by category...")
    plot_influence_change_bars(all_change_data)

    print("  3. Scatter FT vs PT...")
    plot_scatter_ft_vs_pt(all_influence_data)

    print("  4. Influence evolution...")
    plot_influence_evolution(all_influence_data)

    print("  5. Broken queries detail...")
    plot_broken_queries_detail(all_influence_data)

    print("  6. Violin distributions...")
    plot_influence_distribution_violin(all_influence_data)

    print(f"\n{'='*70}")
    print(f"All figures saved to: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
