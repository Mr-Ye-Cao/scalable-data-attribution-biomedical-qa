#!/usr/bin/env python3
"""
GOAL3 Experiment: BM25 Baseline for Training Data Attribution

Compares BM25 retrieval against RapidIn for attribution accuracy.
Indexes training data (finetune + pretrain) and retrieves top-K for test queries.

Usage:
    python scripts/attribution/goal3/bm25_baseline.py --top-k 10
    python scripts/attribution/goal3/bm25_baseline.py --top-k 10 --test-only-correct

Input:
    - Training: results/controlled_experiment/data/finetune_500_raw.jsonl
    - Training: results/controlled_experiment/data/pretrain_500_aligned.jsonl (or pretrain_500_label_matched.jsonl)
    - Test: pubmedqa/data/ori_pqal.json + test_ground_truth.json

Output:
    - results/goal3/bm25_attribution_top{k}.jsonl
"""

import argparse
import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

BASE_DIR = Path("/lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune")

# Use the same data as RapidIn analysis (from data-aggregate)
DATA_AGG = BASE_DIR / "data-aggregate/input"
FINETUNE_DATA = DATA_AGG / "finetune/finetune_500.jsonl"
PRETRAIN_DATA = DATA_AGG / "pretrain/pretrain_500_iter4.jsonl"  # Same as RapidIn report
TEST_DATA = DATA_AGG / "test/test_500.jsonl"
OUTPUT_DIR = BASE_DIR / "results/goal3"


def extract_text_from_instruction(instruction: str) -> str:
    """Extract clean text from QA-formatted instruction."""
    text = instruction
    if '<|im_start|>' in instruction:
        # Remove template markers for cleaner text
        text = text.replace('<|im_start|>', ' ').replace('<|im_end|>', ' ')
        text = text.replace('system', '').replace('user', '').replace('assistant', '')
    return text.strip()


def load_finetune_data() -> list[dict]:
    """Load fine-tuning data with source label.

    Format: {instruction, output, pubid, question, final_decision}
    """
    data = []
    with open(FINETUNE_DATA) as f:
        for line in f:
            sample = json.loads(line)
            # Extract text from instruction (QA formatted)
            text = extract_text_from_instruction(sample['instruction'])
            data.append({
                'id': sample['pubid'],
                'source': 'finetune',
                'text': text,
                'question': sample.get('question', ''),
                'context': text[:500],
                'final_decision': sample['final_decision'],
            })
    return data


def load_pretrain_data() -> list[dict]:
    """Load pre-training data with source label.

    Format: {instruction, output, doc_id, question, final_decision, source_pubid, bm25_score}
    """
    data = []
    with open(PRETRAIN_DATA) as f:
        for line in f:
            sample = json.loads(line)
            # Extract text from instruction (QA formatted)
            text = extract_text_from_instruction(sample['instruction'])
            data.append({
                'id': sample.get('doc_id', f"pretrain_{len(data)}"),
                'source': 'pretrain',
                'text': text,
                'question': sample.get('question', ''),
                'context': text[:500],
                'final_decision': sample.get('final_decision', ''),
            })
    return data


def load_test_queries() -> list[dict]:
    """Load test queries from data-aggregate.

    Format: {instruction, output, pubid, question, ground_truth}
    """
    queries = []
    with open(TEST_DATA) as f:
        for line in f:
            sample = json.loads(line)
            # Extract text from instruction (QA formatted)
            text = extract_text_from_instruction(sample['instruction'])
            queries.append({
                'pubid': sample['pubid'],
                'question': sample.get('question', ''),
                'context': text,
                'final_decision': sample.get('ground_truth', ''),  # Use ground_truth
                'ground_truth': sample.get('ground_truth', ''),
            })
    return queries


def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenization with lowercasing."""
    return text.lower().split()


def build_query_text(query: dict) -> str:
    """Build query text from test sample."""
    return f"{query['context']} {query['question']}"


def run_bm25_attribution(
    queries: list[dict],
    corpus: list[dict],
    top_k: int = 10
) -> tuple[list[dict], dict]:
    """
    Run BM25 retrieval for attribution.

    Returns:
        results: List of attribution results per query
        stats: Summary statistics
    """
    print(f"\nBuilding BM25 index from {len(corpus)} training samples...")
    start = time.time()

    # Tokenize corpus
    tokenized_corpus = [tokenize(doc['text']) for doc in tqdm(corpus, desc="Tokenizing")]

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_corpus)
    index_time = time.time() - start
    print(f"Index built in {index_time:.2f}s")

    # Search
    print(f"\nSearching {len(queries)} queries (top-{top_k})...")
    results = []
    retrieval_times = []

    for query in tqdm(queries, desc="Searching"):
        query_text = build_query_text(query)
        tokenized_query = tokenize(query_text)

        # Time individual retrieval
        t0 = time.time()
        scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        retrieval_times.append(time.time() - t0)

        # Build candidates with source tracking
        candidates = []
        for rank, idx in enumerate(top_indices):
            doc = corpus[idx]
            candidates.append({
                'rank': rank + 1,
                'score': float(scores[idx]),
                'doc_id': doc['id'],
                'source': doc['source'],
                'question': doc['question'][:100],
                'final_decision': doc['final_decision'],
            })

        # Count sources in top-K
        source_counts = Counter(c['source'] for c in candidates)

        results.append({
            'pubid': query['pubid'],
            'question': query['question'],
            'context': query['context'][:200] + "...",
            'final_decision': query['final_decision'],
            'ground_truth': query['ground_truth'],
            'candidates': candidates,
            'finetune_count': source_counts.get('finetune', 0),
            'pretrain_count': source_counts.get('pretrain', 0),
            'finetune_ratio': source_counts.get('finetune', 0) / top_k,
        })

    # Compute statistics
    stats = {
        'index_time_sec': index_time,
        'avg_retrieval_time_ms': np.mean(retrieval_times) * 1000,
        'total_retrieval_time_sec': sum(retrieval_times),
        'corpus_size': len(corpus),
        'num_queries': len(queries),
        'top_k': top_k,
    }

    return results, stats


def main():
    parser = argparse.ArgumentParser(description="BM25 baseline for GOAL3 attribution experiment")
    parser.add_argument('--top-k', type=int, default=10, help='Number of candidates per query')
    parser.add_argument('--test-only-correct', action='store_true',
                        help='Only evaluate on queries where model prediction matches ground truth')
    parser.add_argument('--pretrain-file', type=str, default='pretrain_500_iter4.jsonl',
                        choices=['pretrain_500_iter1.jsonl', 'pretrain_500_iter2.jsonl',
                                 'pretrain_500_iter3.jsonl', 'pretrain_500_iter4.jsonl',
                                 'pretrain_500_noformat.jsonl'],
                        help='Which pretrain file to use (default: iter4 to match RapidIn report)')
    args = parser.parse_args()

    print("=" * 70)
    print("GOAL3: BM25 Baseline for Training Data Attribution")
    print("=" * 70)

    # Update pretrain path if specified
    global PRETRAIN_DATA
    PRETRAIN_DATA = DATA_AGG / "pretrain" / args.pretrain_file

    # Load training corpus
    print(f"\nLoading training corpus...")
    finetune_data = load_finetune_data()
    print(f"  Finetune samples: {len(finetune_data)}")

    pretrain_data = load_pretrain_data()
    print(f"  Pretrain samples: {len(pretrain_data)}")

    corpus = finetune_data + pretrain_data
    print(f"  Total corpus: {len(corpus)}")

    # Load test queries
    print(f"\nLoading test queries...")
    queries = load_test_queries()
    print(f"  Test queries: {len(queries)}")

    if args.test_only_correct:
        # TODO: Filter to only correctly answered queries
        # This requires model predictions - skip for now
        print("  (--test-only-correct not yet implemented, using all queries)")

    # Run BM25 attribution
    results, stats = run_bm25_attribution(queries, corpus, top_k=args.top_k)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"bm25_attribution_top{args.top_k}.jsonl"

    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    print(f"\nSaved to: {output_file}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("EFFICIENCY METRICS (for GOAL3 Experiment 1)")
    print("=" * 70)
    print(f"Corpus size: {stats['corpus_size']} samples")
    print(f"Index construction time: {stats['index_time_sec']:.2f} seconds")
    print(f"Average retrieval time: {stats['avg_retrieval_time_ms']:.2f} ms/query")
    print(f"Total retrieval time: {stats['total_retrieval_time_sec']:.2f} seconds")
    print(f"Throughput: {stats['num_queries'] / stats['total_retrieval_time_sec']:.1f} queries/sec")

    print("\n" + "=" * 70)
    print("ATTRIBUTION METRICS (for GOAL3 Experiment 2)")
    print("=" * 70)

    # Compute aggregate statistics
    finetune_ratios = [r['finetune_ratio'] for r in results]
    print(f"Top-K: {args.top_k}")
    print(f"Average finetune ratio in top-{args.top_k}: {np.mean(finetune_ratios):.3f}")
    print(f"Median finetune ratio: {np.median(finetune_ratios):.3f}")
    print(f"Queries with majority finetune: {sum(1 for r in finetune_ratios if r > 0.5)} / {len(results)}")
    print(f"Queries with any finetune in top-{args.top_k}: {sum(1 for r in finetune_ratios if r > 0)} / {len(results)}")

    # Breakdown by ground truth label
    print("\n--- Breakdown by Ground Truth Label ---")
    for label in ['yes', 'no', 'maybe']:
        label_results = [r for r in results if r['ground_truth'] == label]
        if label_results:
            avg_ratio = np.mean([r['finetune_ratio'] for r in label_results])
            print(f"  {label}: {len(label_results)} queries, avg finetune ratio = {avg_ratio:.3f}")

    # Sample results
    print("\n" + "=" * 70)
    print("SAMPLE RESULTS (first 3 queries)")
    print("=" * 70)

    for r in results[:3]:
        print(f"\nQuery: {r['question'][:60]}...")
        print(f"  Ground truth: {r['ground_truth']}")
        print(f"  Finetune in top-{args.top_k}: {r['finetune_count']}, Pretrain: {r['pretrain_count']}")
        if r['candidates']:
            c = r['candidates'][0]
            print(f"  Top-1 (score={c['score']:.2f}, source={c['source']}): {c['question'][:50]}...")

    # Save stats
    stats_file = OUTPUT_DIR / f"bm25_stats_top{args.top_k}.json"
    with open(stats_file, 'w') as f:
        json.dump({
            **stats,
            'avg_finetune_ratio': float(np.mean(finetune_ratios)),
            'median_finetune_ratio': float(np.median(finetune_ratios)),
        }, f, indent=2)
    print(f"\nStats saved to: {stats_file}")

    print("\n" + "=" * 70)
    print("BM25 Baseline Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
