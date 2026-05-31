#!/usr/bin/env python3
"""
Export influence scores from gradient files to JSON.
Creates per-test-query influence scores for data-aggregate folder.
"""

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("/lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune")

# Gradients paths
TEST_GRADS_DIR = BASE_DIR / "results/rapidin_aligned_500/grads"
FT_GRADS_DIR = BASE_DIR / "results/rapidin_aligned_500/grads"
PT_GRADS_DIR = BASE_DIR / "results/validation/grads"

OUTPUT_DIR = BASE_DIR / "data-aggregate/outputs/influence_scores"

MODELS = ["baseline", "ckpt32", "ckpt64"]
DATASETS = ["finetune", "noformat", "iter1", "iter2", "iter3", "iter4"]


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


def main():
    print("=" * 80)
    print("EXPORTING INFLUENCE SCORES")
    print(f"Time: {datetime.now()}")
    print("=" * 80)

    for model in MODELS:
        print(f"\n{'#' * 60}")
        print(f"# MODEL: {model.upper()}")
        print(f"{'#' * 60}")

        model_dir = OUTPUT_DIR / model
        model_dir.mkdir(parents=True, exist_ok=True)

        # Load test gradients
        test_grads = load_gradients(TEST_GRADS_DIR / model / "test")
        if test_grads is None:
            print(f"Missing test gradients for {model}")
            continue
        print(f"Loaded {len(test_grads)} test gradients")

        for dataset in DATASETS:
            print(f"\n--- {dataset} ---")

            # Determine gradient path
            if dataset == "finetune":
                train_grads = load_gradients(FT_GRADS_DIR / model / "finetune")
            elif dataset == "noformat":
                train_grads = load_gradients(PT_GRADS_DIR / model / "noformat")
            else:
                # Try both iter1 and pretrain_iter1 naming conventions
                train_grads = load_gradients(PT_GRADS_DIR / model / dataset)
                if train_grads is None:
                    train_grads = load_gradients(PT_GRADS_DIR / model / f"pretrain_{dataset}")

            if train_grads is None:
                print(f"  Missing train gradients for {model}/{dataset}")
                continue

            print(f"  Loaded {len(train_grads)} train gradients")

            # Compute influence scores
            influence_matrix = compute_influence(test_grads, train_grads).numpy()
            print(f"  Influence matrix shape: {influence_matrix.shape}")

            # Extract per-query max influence and top-k indices
            max_influences = influence_matrix.max(axis=1)
            top_k_indices = influence_matrix.argsort(axis=1)[:, -10:][:, ::-1]  # Top 10
            top_k_scores = np.take_along_axis(influence_matrix, top_k_indices, axis=1)

            # Create output structure
            output = {
                "model": model,
                "dataset": dataset,
                "n_test": int(influence_matrix.shape[0]),
                "n_train": int(influence_matrix.shape[1]),
                "summary": {
                    "mean_max": float(np.mean(max_influences)),
                    "std_max": float(np.std(max_influences)),
                    "min_max": float(np.min(max_influences)),
                    "max_max": float(np.max(max_influences)),
                    "median_max": float(np.median(max_influences)),
                },
                "per_query": []
            }

            # Per-query details
            for i in range(len(test_grads)):
                output["per_query"].append({
                    "test_idx": i,
                    "max_influence": float(max_influences[i]),
                    "top_10_indices": top_k_indices[i].tolist(),
                    "top_10_scores": top_k_scores[i].tolist()
                })

            # Save to JSON
            output_file = model_dir / f"{dataset}.json"
            with open(output_file, "w") as f:
                json.dump(output, f, indent=2)
            print(f"  Saved to {output_file}")
            print(f"  Mean max influence: {output['summary']['mean_max']:.4f}")

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
