#!/usr/bin/env python3
"""
Step 3 of Controlled Attribution Experiment:
Build pre-training subset from BM25 results by selecting documents with highest overlap.

Strategy: Select documents that appear as candidates for the most fine-tuning queries.
This gives us the most "concentrated" pre-training samples that are relevant to multiple queries.

Usage:
    python scripts/attribution/controlled_experiment/build_pretrain_subset.py --top-n 500

Input: results/controlled_experiment/data/bm25_results_0013_top10.jsonl
Output: results/controlled_experiment/data/pretrain_500_raw.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

BASE_DIR = Path("/lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune")
BM25_RESULTS = BASE_DIR / "results/controlled_experiment/data/bm25_results_0013_top10.jsonl"
OUTPUT_DIR = BASE_DIR / "results/controlled_experiment/data"


def load_bm25_results() -> list[dict]:
    """Load BM25 search results."""
    results = []
    with open(BM25_RESULTS) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def count_document_frequencies(results: list[dict]) -> Counter:
    """Count how many queries each document appears in."""
    doc_freq = Counter()

    for r in results:
        # Count each document once per query (even if it appears multiple times)
        seen_docs = set()
        for candidate in r['candidates']:
            doc_id = candidate['doc_id']
            if doc_id not in seen_docs:
                doc_freq[doc_id] += 1
                seen_docs.add(doc_id)

    return doc_freq


def get_document_details(results: list[dict]) -> dict:
    """Build a lookup of document details (text, shard, best score)."""
    doc_details = {}

    for r in results:
        for candidate in r['candidates']:
            doc_id = candidate['doc_id']
            if doc_id not in doc_details:
                doc_details[doc_id] = {
                    'doc_id': doc_id,
                    'text': candidate['text'],
                    'shard': candidate['shard'],
                    'best_score': candidate['score'],
                    'best_rank': candidate['rank'],
                }
            else:
                # Update if this appearance has better score
                if candidate['score'] > doc_details[doc_id]['best_score']:
                    doc_details[doc_id]['best_score'] = candidate['score']
                    doc_details[doc_id]['best_rank'] = candidate['rank']

    return doc_details


def build_query_mapping(results: list[dict], selected_doc_ids: set) -> dict:
    """Build mapping of which queries each selected document is relevant to."""
    doc_to_queries = {doc_id: [] for doc_id in selected_doc_ids}

    for r in results:
        for candidate in r['candidates']:
            doc_id = candidate['doc_id']
            if doc_id in selected_doc_ids:
                doc_to_queries[doc_id].append({
                    'pubid': r['pubid'],
                    'question': r['question'][:100],
                    'rank': candidate['rank'],
                    'score': candidate['score'],
                })

    return doc_to_queries


def main():
    parser = argparse.ArgumentParser(description="Build pre-training subset from BM25 results")
    parser.add_argument('--top-n', type=int, default=500, help='Number of documents to select')
    parser.add_argument('--min-freq', type=int, default=1, help='Minimum frequency threshold')
    args = parser.parse_args()

    print("=" * 70)
    print("Step 3: Build Pre-training Subset from BM25 Results")
    print("=" * 70)

    # Load BM25 results
    print(f"\nLoading BM25 results from {BM25_RESULTS}...")
    results = load_bm25_results()
    print(f"Loaded {len(results)} query results")

    # Count document frequencies
    print("\nCounting document frequencies across queries...")
    doc_freq = count_document_frequencies(results)
    print(f"Total unique documents: {len(doc_freq)}")

    # Frequency distribution
    freq_dist = Counter(doc_freq.values())
    print("\nDocument frequency distribution:")
    for freq in sorted(freq_dist.keys(), reverse=True)[:10]:
        print(f"  Appears in {freq} queries: {freq_dist[freq]} documents")

    # Select top-N documents by frequency
    print(f"\nSelecting top-{args.top_n} documents by frequency...")
    top_docs = doc_freq.most_common(args.top_n)
    selected_doc_ids = set(doc_id for doc_id, _ in top_docs)

    print(f"Selected {len(selected_doc_ids)} documents")
    print(f"  Min frequency in selection: {top_docs[-1][1]} queries")
    print(f"  Max frequency in selection: {top_docs[0][1]} queries")

    # Get document details
    print("\nGathering document details...")
    doc_details = get_document_details(results)

    # Build query mapping
    print("Building query mapping...")
    doc_to_queries = build_query_mapping(results, selected_doc_ids)

    # Create output records
    output_records = []
    for doc_id, freq in top_docs:
        details = doc_details[doc_id]
        record = {
            'doc_id': doc_id,
            'text': details['text'],
            'shard': details['shard'],
            'frequency': freq,  # Number of queries this doc is relevant to
            'best_score': details['best_score'],
            'best_rank': details['best_rank'],
            'relevant_queries': doc_to_queries[doc_id],
        }
        output_records.append(record)

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"pretrain_{args.top_n}_raw.jsonl"

    with open(output_file, 'w') as f:
        for record in output_records:
            f.write(json.dumps(record) + '\n')

    print(f"\nSaved to: {output_file}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_freq = sum(freq for _, freq in top_docs)
    avg_freq = total_freq / len(top_docs)

    print(f"Selected documents: {len(output_records)}")
    print(f"Total query coverage: {total_freq} (sum of frequencies)")
    print(f"Average frequency: {avg_freq:.2f} queries per document")

    # Coverage analysis
    covered_queries = set()
    for record in output_records:
        for q in record['relevant_queries']:
            covered_queries.add(q['pubid'])
    print(f"Queries covered: {len(covered_queries)}/{len(results)} ({100*len(covered_queries)/len(results):.1f}%)")

    # Text length statistics
    text_lengths = [len(r['text']) for r in output_records]
    print(f"\nDocument text lengths:")
    print(f"  Mean: {sum(text_lengths)/len(text_lengths):.0f} chars")
    print(f"  Min: {min(text_lengths)} chars")
    print(f"  Max: {max(text_lengths)} chars")

    # Sample output
    print("\n" + "=" * 70)
    print("SAMPLE OUTPUT (top 3 most frequent documents)")
    print("=" * 70)

    for record in output_records[:3]:
        print(f"\nDoc ID: {record['doc_id']}")
        print(f"  Frequency: {record['frequency']} queries")
        print(f"  Best score: {record['best_score']:.2f} (rank {record['best_rank']})")
        print(f"  Text preview: {record['text'][:100]}...")
        print(f"  Sample relevant queries:")
        for q in record['relevant_queries'][:2]:
            print(f"    - {q['question'][:60]}... (rank {q['rank']}, score {q['score']:.1f})")

    print("\n" + "=" * 70)
    print("Step 3 Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
