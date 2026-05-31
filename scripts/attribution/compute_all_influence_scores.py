#!/usr/bin/env python3
"""
Compute all influence scores for new checkpoints (ckpt96, ckpt128, ckpt160).
Datasets: finetune, iter1, iter4, noformat
"""

import json
import torch
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("/lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune")

# Checkpoint configurations
CHECKPOINTS = ["ckpt96", "ckpt128", "ckpt160"]

# Dataset paths
DATASETS = {
    "finetune": "results/rapidin_aligned_500/grads/{ckpt}/finetune",
    "iter1": "results/validation/grads/{ckpt}/iter1",
    "iter4": "results/validation/grads/{ckpt}/iter4",
    "noformat": "results/validation/grads/{ckpt}/noformat",
}

TEST_GRADS_PATH = "results/rapidin_aligned_500/grads/{ckpt}/test"
OUTPUT_DIR = BASE_DIR / "data-aggregate/outputs/influence_scores"


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
    # For each test query, find max influence and top-10
    max_per_query = influence_matrix.max(dim=1).values
    top_10_per_query = torch.topk(influence_matrix, k=min(10, influence_matrix.shape[1]), dim=1)

    return {
        "mean_max": float(max_per_query.mean()),
        "std_max": float(max_per_query.std()),
        "min_max": float(max_per_query.min()),
        "max_max": float(max_per_query.max()),
        "median_max": float(max_per_query.median()),
        "max_per_query": max_per_query.tolist(),
        "top_10_indices": top_10_per_query.indices.tolist(),
        "top_10_scores": top_10_per_query.values.tolist(),
    }


def main():
    print("=" * 80)
    print("COMPUTING INFLUENCE SCORES FOR NEW CHECKPOINTS")
    print("=" * 80)
    print(f"\nTime: {datetime.now()}")

    for ckpt in CHECKPOINTS:
        print(f"\n{'=' * 60}")
        print(f"PROCESSING: {ckpt.upper()}")
        print("=" * 60)

        # Load test gradients
        test_path = BASE_DIR / TEST_GRADS_PATH.format(ckpt=ckpt)
        print(f"\nLoading test gradients from {test_path}...")
        test_grads = load_gradients(test_path)
        if test_grads is None:
            print(f"ERROR: Cannot load test gradients for {ckpt}!")
            continue
        print(f"  Test gradients shape: {test_grads.shape}")

        # Process each dataset
        for dataset_name, dataset_path_template in DATASETS.items():
            print(f"\n{'-' * 40}")
            print(f"{dataset_name.upper()}:")

            dataset_path = BASE_DIR / dataset_path_template.format(ckpt=ckpt)
            train_grads = load_gradients(dataset_path)

            if train_grads is None:
                print(f"  Skipping {dataset_name} - gradients not found")
                continue

            print(f"  Shape: {train_grads.shape}")

            # Compute influence
            influence = compute_influence(test_grads, train_grads)
            stats = compute_statistics(influence)

            print(f"  Mean Max Influence: {stats['mean_max']:.4f}")
            print(f"  Std: {stats['std_max']:.4f}")

            # Save results
            output_dir = OUTPUT_DIR / ckpt
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{dataset_name}.json"

            output_data = {
                "model": ckpt,
                "dataset": dataset_name,
                "n_test": test_grads.shape[0],
                "n_train": train_grads.shape[0],
                "summary": {
                    "mean_max": stats["mean_max"],
                    "std_max": stats["std_max"],
                    "min_max": stats["min_max"],
                    "max_max": stats["max_max"],
                    "median_max": stats["median_max"]
                },
                "per_query": [
                    {
                        "test_idx": i,
                        "max_influence": stats["max_per_query"][i],
                        "top_10_indices": stats["top_10_indices"][i],
                        "top_10_scores": stats["top_10_scores"][i]
                    }
                    for i in range(len(stats["max_per_query"]))
                ]
            }

            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"  Saved: {output_file}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
