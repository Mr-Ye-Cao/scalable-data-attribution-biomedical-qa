#!/usr/bin/env python3
"""
Step 2 of Controlled Attribution Experiment:
Run BM25 semantic search to find relevant pre-training documents for fine-tuning samples.

Usage:
    python scripts/attribution/controlled_experiment/bm25_retrieve.py --subset 0013 --top-k 10
    python scripts/attribution/controlled_experiment/bm25_retrieve.py --all-subsets --top-k 10

Input: results/controlled_experiment/data/finetune_500_raw.jsonl
Output: results/controlled_experiment/data/bm25_results_top{k}.jsonl
"""

import argparse
import json
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import zstandard as zstd
from rank_bm25 import BM25Okapi
from tqdm import tqdm

BASE_DIR = Path("/lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune")
DOLMA_DIR = BASE_DIR / "data/dolma3_health"
FINETUNE_RAW = BASE_DIR / "results/controlled_experiment/data/finetune_500_raw.jsonl"
OUTPUT_DIR = BASE_DIR / "results/controlled_experiment/data"


def load_zst_shard(shard_path: Path) -> Iterator[dict]:
    """Load a single .jsonl.zst shard, yielding documents one by one."""
    dctx = zstd.ZstdDecompressor()
    with open(shard_path, 'rb') as f:
        with dctx.stream_reader(f) as reader:
            text_stream = reader.read().decode('utf-8')
            for line in text_stream.strip().split('\n'):
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def iter_all_shards(subsets: list[str] = None) -> Iterator[Path]:
    """Iterate over all shard files."""
    if subsets:
        subset_dirs = [DOLMA_DIR / f"common_crawl-health-{s}" for s in subsets]
    else:
        subset_dirs = sorted(DOLMA_DIR.glob("common_crawl-health-*"))

    for subset_dir in subset_dirs:
        if not subset_dir.is_dir():
            continue
        shard_files = sorted(subset_dir.glob("*.jsonl.zst"))
        for shard_file in shard_files:
            yield shard_file


def load_corpus(subsets: list[str] = None, max_docs: int = None, min_length: int = 100) -> tuple[list[str], list[dict]]:
    """Load pre-training corpus from Dolma3 health shards."""
    documents = []
    metadata = []

    shard_files = list(iter_all_shards(subsets))
    print(f"Found {len(shard_files)} shard files")

    for shard_path in tqdm(shard_files, desc="Loading corpus"):
        for doc in load_zst_shard(shard_path):
            text = doc.get('text', '')
            if len(text) < min_length:
                continue

            # Limit document length for efficiency
            documents.append(text[:5000])
            metadata.append({
                'id': doc.get('id', f'doc_{len(metadata)}'),
                'text': text,  # Keep full text for later use
                'shard': str(shard_path.relative_to(DOLMA_DIR)),
            })

            if max_docs and len(documents) >= max_docs:
                return documents, metadata

    return documents, metadata


def load_finetune_queries() -> list[dict]:
    """Load raw fine-tuning data as queries."""
    queries = []
    with open(FINETUNE_RAW) as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def build_query_text(sample: dict) -> str:
    """Build query text from raw fine-tuning sample."""
    # Combine context + question for richer query
    return f"{sample['context']} {sample['question']}"


def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenization with lowercasing."""
    return text.lower().split()


def run_bm25_search(queries: list[dict], documents: list[str], metadata: list[dict], top_k: int = 10) -> list[dict]:
    """Run BM25 search for all queries."""
    print(f"\nBuilding BM25 index from {len(documents)} documents...")
    start = time.time()

    # Tokenize corpus
    tokenized_corpus = [tokenize(doc) for doc in tqdm(documents, desc="Tokenizing")]

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"Index built in {time.time() - start:.1f}s")

    # Search
    print(f"\nSearching {len(queries)} queries (top-{top_k})...")
    results = []

    for query in tqdm(queries, desc="Searching"):
        query_text = build_query_text(query)
        tokenized_query = tokenize(query_text)

        # Get BM25 scores
        scores = bm25.get_scores(tokenized_query)

        # Get top-k
        top_indices = np.argsort(scores)[-top_k:][::-1]

        candidates = []
        for rank, idx in enumerate(top_indices):
            m = metadata[idx]
            candidates.append({
                'rank': rank + 1,
                'score': float(scores[idx]),
                'doc_id': m['id'],
                'shard': m['shard'],
                'text': m['text'],  # Keep full text for step 4
            })

        results.append({
            'pubid': query['pubid'],
            'question': query['question'],
            'context': query['context'][:200] + "...",  # Preview only
            'final_decision': query['final_decision'],
            'candidates': candidates,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="BM25 retrieval for controlled experiment")
    parser.add_argument('--subset', type=str, default='0013', help='Single subset to use (e.g., 0013)')
    parser.add_argument('--all-subsets', action='store_true', help='Use all available subsets')
    parser.add_argument('--top-k', type=int, default=10, help='Number of candidates per query')
    parser.add_argument('--max-docs', type=int, default=None, help='Limit corpus size (for testing)')
    args = parser.parse_args()

    print("=" * 70)
    print("Step 2: BM25 Retrieval for Controlled Attribution Experiment")
    print("=" * 70)

    # Determine subsets
    if args.all_subsets:
        subsets = None  # Will use all
        subset_str = "all"
    else:
        subsets = [args.subset]
        subset_str = args.subset

    # Load corpus
    print(f"\nLoading corpus from Dolma3 health (subset={subset_str})...")
    documents, metadata = load_corpus(subsets, max_docs=args.max_docs)
    print(f"Loaded {len(documents)} documents")

    # Load queries
    print(f"\nLoading fine-tuning queries from {FINETUNE_RAW}...")
    queries = load_finetune_queries()
    print(f"Loaded {len(queries)} queries")

    # Run BM25 search
    results = run_bm25_search(queries, documents, metadata, top_k=args.top_k)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"bm25_results_{subset_str}_top{args.top_k}.jsonl"

    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    print(f"\nSaved to: {output_file}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    scores = [r['candidates'][0]['score'] if r['candidates'] else 0 for r in results]
    print(f"Queries: {len(results)}")
    print(f"Corpus size: {len(documents)}")
    print(f"Top-k: {args.top_k}")
    print(f"Average top-1 BM25 score: {np.mean(scores):.4f}")
    print(f"Min top-1 score: {np.min(scores):.4f}")
    print(f"Max top-1 score: {np.max(scores):.4f}")

    # Sample results
    print("\n" + "=" * 70)
    print("SAMPLE RESULTS (first 3 queries)")
    print("=" * 70)

    for r in results[:3]:
        print(f"\nQuery: {r['question'][:60]}...")
        print(f"  Decision: {r['final_decision']}")
        if r['candidates']:
            c = r['candidates'][0]
            print(f"  Top-1 (score={c['score']:.4f}): {c['text'][:100]}...")

    print("\n" + "=" * 70)
    print("Step 2 Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
