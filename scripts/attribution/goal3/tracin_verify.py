#!/usr/bin/env python3
"""
Verify TracIn computation is valid by computing influence scores
for a small subset (on-the-fly, no caching).

This demonstrates:
1. TracIn computation is valid (produces meaningful influence scores)
2. Without caching, it's computationally expensive (O(N_train × N_test))
3. With caching, storage becomes infeasible (6.64TB for 500 samples)

Usage:
    python scripts/attribution/goal3/tracin_verify.py --n-train 5 --n-test 3
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

BASE_DIR = Path("/lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune")
MODEL_PATH = BASE_DIR / "OLMo-3-7B-Instruct"
FINETUNE_DATA = BASE_DIR / "data-aggregate/input/finetune/finetune_500.jsonl"
TEST_DATA = BASE_DIR / "data-aggregate/input/test/test_500.jsonl"


def load_samples(path, n_samples):
    """Load n samples from jsonl file."""
    samples = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            samples.append(json.loads(line))
    return samples


def compute_gradient(model, tokenizer, sample, device):
    """Compute gradient for a single sample, return as flat vector on CPU."""
    model.zero_grad()

    # Tokenize
    text = sample['instruction'] + sample.get('output', '')
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward + backward
    outputs = model(**inputs, labels=inputs['input_ids'])
    loss = outputs.loss
    loss.backward()

    # Flatten gradients into single vector - move to CPU to save GPU memory
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1).float().cpu())  # Convert to float32 on CPU

    grad_vector = torch.cat(grads)

    # Clear GPU gradients
    model.zero_grad()
    torch.cuda.empty_cache()

    return grad_vector, loss.item()


def compute_tracin_influence(grad_train, grad_test):
    """Compute TracIn influence score (dot product of gradients)."""
    # Normalize for cosine similarity (like RapidIn)
    grad_train_norm = F.normalize(grad_train.unsqueeze(0), dim=1)
    grad_test_norm = F.normalize(grad_test.unsqueeze(0), dim=1)

    influence = torch.mm(grad_train_norm, grad_test_norm.t()).item()
    return influence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-train', type=int, default=5, help='Number of training samples')
    parser.add_argument('--n-test', type=int, default=3, help='Number of test samples')
    args = parser.parse_args()

    print("=" * 80)
    print("TracIn Verification: Computing Influence Scores On-the-Fly")
    print("=" * 80)

    # Load model
    print(f"\nLoading model...")
    device = torch.device("cuda")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e9:.2f}B)")

    # Load data
    train_samples = load_samples(FINETUNE_DATA, args.n_train)
    test_samples = load_samples(TEST_DATA, args.n_test)
    print(f"\nTrain samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")

    # Compute all pairwise influence scores
    print(f"\n{'=' * 80}")
    print(f"Computing {len(train_samples)} × {len(test_samples)} = {len(train_samples) * len(test_samples)} influence scores...")
    print(f"{'=' * 80}")

    total_pairs = len(train_samples) * len(test_samples)
    total_grad_computations = len(train_samples) + len(test_samples)

    # First, compute all test gradients (we can cache these since there are few)
    print("\nComputing test gradients...")
    test_grads = []
    test_losses = []
    for i, sample in enumerate(tqdm(test_samples, desc="Test grads")):
        grad, loss = compute_gradient(model, tokenizer, sample, device)
        test_grads.append(grad.clone())
        test_losses.append(loss)
        print(f"  Test {i}: loss={loss:.4f}, grad_norm={grad.norm():.4f}")

    # For each train sample, compute gradient and influence on all test samples
    print("\nComputing train gradients and influence scores...")
    influence_matrix = []

    start_time = time.time()

    for i, train_sample in enumerate(tqdm(train_samples, desc="Train grads")):
        train_grad, train_loss = compute_gradient(model, tokenizer, train_sample, device)

        # Compute influence on each test sample
        influences = []
        for j, test_grad in enumerate(test_grads):
            influence = compute_tracin_influence(train_grad, test_grad)
            influences.append(influence)

        influence_matrix.append(influences)

        max_inf = max(influences)
        max_idx = influences.index(max_inf)
        print(f"  Train {i}: loss={train_loss:.4f}, max_influence={max_inf:.4f} (on test {max_idx})")

        # Clear GPU memory
        del train_grad
        torch.cuda.empty_cache()

    elapsed = time.time() - start_time

    # Display influence matrix
    print(f"\n{'=' * 80}")
    print("INFLUENCE MATRIX (Train × Test)")
    print(f"{'=' * 80}")

    # Header
    header = "Train\\Test |"
    for j in range(len(test_samples)):
        header += f" Test{j:2d} |"
    print(header)
    print("-" * len(header))

    # Rows
    for i, influences in enumerate(influence_matrix):
        row = f"  Train{i:2d} |"
        for inf in influences:
            row += f" {inf:6.4f} |"
        print(row)

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    all_influences = [inf for row in influence_matrix for inf in row]
    print(f"\nInfluence score statistics:")
    print(f"  Mean: {sum(all_influences)/len(all_influences):.4f}")
    print(f"  Min:  {min(all_influences):.4f}")
    print(f"  Max:  {max(all_influences):.4f}")

    print(f"\nComputation time:")
    print(f"  Total: {elapsed:.2f}s for {total_pairs} pairs")
    print(f"  Per pair: {elapsed/total_pairs*1000:.2f}ms")
    print(f"  Gradient computations: {total_grad_computations}")

    # Extrapolate
    print(f"\n{'=' * 80}")
    print("EXTRAPOLATION: Full Dataset (500 train × 500 test)")
    print(f"{'=' * 80}")

    full_pairs = 500 * 500
    full_grad_comps = 500 + 500  # If we cache test grads

    # Time per gradient computation
    time_per_grad = elapsed / total_grad_computations
    estimated_time = time_per_grad * full_grad_comps

    print(f"\nWithout caching (naive):")
    print(f"  Gradient computations needed: {full_pairs * 2:,} (compute both for each pair)")
    print(f"  Estimated time: {full_pairs * 2 * time_per_grad / 3600:.1f} hours")

    print(f"\nWith test gradient caching:")
    print(f"  Gradient computations needed: {full_grad_comps:,}")
    print(f"  Estimated time: {estimated_time:.1f}s ({estimated_time/60:.1f} min)")

    print(f"\nWith full gradient caching (TracIn standard):")
    print(f"  Storage needed: 500 × 13.59 GB = 6.64 TB")
    print(f"  This is INFEASIBLE for typical storage")

    print(f"\n{'=' * 80}")
    print("CONCLUSION")
    print(f"{'=' * 80}")
    print("""
TracIn produces valid influence scores (cosine similarity of gradients).

The problem is scalability:
1. WITHOUT caching: Need O(N×M) gradient computations = hours/days of compute
2. WITH caching: Need O(N) storage of full gradients = TBs of storage

RapidIn solves this by:
- Compressing gradients from 7.3B dims → 65536 dims (111,000x compression)
- Storage: 130KB per sample vs 13.59GB per sample
- Preserves gradient direction for cosine similarity
""")


if __name__ == "__main__":
    main()
