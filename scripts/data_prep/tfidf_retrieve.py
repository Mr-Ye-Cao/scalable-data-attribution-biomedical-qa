#!/usr/bin/env python3
"""
Fast TF-IDF retrieval using scipy sparse matrices.

Usage:
    python scripts/tfidf_retrieve.py --subset 0013 --top-k 1000
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import zstandard as zstd
from datasets import load_from_disk
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


DATA_DIR = Path("./data/dolma3_health")
INDEX_DIR = Path("./data/tfidf_index")
PUBMEDQA_PATH = Path("./data/pubmedqa/pqa_labeled")
OUTPUT_DIR = Path("./data/bm25_candidates")


def load_shard(shard_path: Path) -> list[dict]:
    """Load a single .jsonl.zst shard."""
    samples = []
    dctx = zstd.ZstdDecompressor()
    with open(shard_path, 'rb') as f:
        with dctx.stream_reader(f) as reader:
            text_stream = reader.read().decode('utf-8')
            for line in text_stream.strip().split('\n'):
                if line:
                    try:
                        sample = json.loads(line)
                        samples.append(sample)
                    except json.JSONDecodeError:
                        continue
    return samples


def iter_all_shards(subset: str = None):
    """Iterate over all shards."""
    if subset:
        subset_dirs = [DATA_DIR / f"common_crawl-health-{subset}"]
    else:
        subset_dirs = sorted(DATA_DIR.glob("common_crawl-health-*"))

    for subset_dir in subset_dirs:
        if not subset_dir.is_dir():
            continue
        shard_files = sorted(subset_dir.glob("*.jsonl.zst"))
        for shard_file in shard_files:
            yield shard_file


def build_index(subset: str = None, max_samples: int = None):
    """Build TF-IDF index."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{subset}" if subset else "_all"

    # Load all documents
    print("Loading documents...")
    documents = []
    metadata = []

    shard_files = list(iter_all_shards(subset))
    print(f"Found {len(shard_files)} shard files")

    for shard_path in tqdm(shard_files, desc="Loading"):
        samples = load_shard(shard_path)
        for sample in samples:
            text = sample.get('text', '')
            if len(text) < 50:
                continue

            documents.append(text[:5000])  # Limit doc length
            metadata.append({
                'id': sample.get('id', f'doc_{len(metadata)}'),
                'text_preview': text[:500],
                'shard': str(shard_path.relative_to(DATA_DIR)),
                'idx': len(metadata)
            })

            if max_samples and len(metadata) >= max_samples:
                break
        if max_samples and len(metadata) >= max_samples:
            break

    print(f"Loaded {len(documents)} documents")

    # Build TF-IDF vectorizer
    print("Building TF-IDF matrix...")
    start = time.time()
    vectorizer = TfidfVectorizer(
        max_features=100000,  # Limit vocabulary
        min_df=2,  # Ignore rare terms
        max_df=0.95,  # Ignore very common terms
        stop_words='english',
        ngram_range=(1, 1),
        dtype=np.float32
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Built in {time.time() - start:.1f}s")

    # Save
    print("Saving index...")
    save_npz(INDEX_DIR / f"tfidf_matrix{suffix}.npz", tfidf_matrix)
    with open(INDEX_DIR / f"vectorizer{suffix}.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(INDEX_DIR / f"metadata{suffix}.jsonl", 'w') as f:
        for m in metadata:
            f.write(json.dumps(m) + '\n')

    print(f"Saved to {INDEX_DIR}")
    return vectorizer, tfidf_matrix, metadata


def load_index(subset: str):
    """Load TF-IDF index."""
    suffix = f"_{subset}" if subset else "_all"

    print("Loading TF-IDF index...")
    tfidf_matrix = load_npz(INDEX_DIR / f"tfidf_matrix{suffix}.npz")
    with open(INDEX_DIR / f"vectorizer{suffix}.pkl", 'rb') as f:
        vectorizer = pickle.load(f)
    metadata = []
    with open(INDEX_DIR / f"metadata{suffix}.jsonl") as f:
        for line in f:
            metadata.append(json.loads(line))

    print(f"Loaded {tfidf_matrix.shape[0]} documents")
    return vectorizer, tfidf_matrix, metadata


def search(vectorizer, tfidf_matrix, metadata, queries: list[str], top_k: int = 1000):
    """Search queries against the index."""
    # Transform queries
    query_vectors = vectorizer.transform(queries)

    # Compute similarities (batch matrix multiplication)
    # shape: (num_queries, num_docs)
    similarities = (query_vectors @ tfidf_matrix.T).toarray()

    results = []
    for i, sim_scores in enumerate(similarities):
        top_indices = np.argsort(sim_scores)[-top_k:][::-1]
        candidates = []
        for rank, idx in enumerate(top_indices):
            m = metadata[idx]
            candidates.append({
                'rank': rank + 1,
                'score': float(sim_scores[idx]),
                'doc_id': m['id'],
                'text_preview': m['text_preview'],
                'shard': m['shard'],
                'idx': m['idx']
            })
        results.append(candidates)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str, default='0013')
    parser.add_argument('--top-k', type=int, default=1000)
    parser.add_argument('--max-queries', type=int, default=None)
    parser.add_argument('--build', action='store_true', help='Build index first')
    args = parser.parse_args()

    suffix = f"_{args.subset}" if args.subset else "_all"

    # Build or load index
    if args.build or not (INDEX_DIR / f"tfidf_matrix{suffix}.npz").exists():
        vectorizer, tfidf_matrix, metadata = build_index(args.subset)
    else:
        vectorizer, tfidf_matrix, metadata = load_index(args.subset)

    # Load PubMedQA
    print(f"Loading PubMedQA...")
    dataset = load_from_disk(str(PUBMEDQA_PATH))['train']
    if args.max_queries:
        dataset = dataset.select(range(min(args.max_queries, len(dataset))))
    print(f"Processing {len(dataset)} queries")

    # Prepare queries
    queries = []
    for sample in dataset:
        question = sample['question']
        contexts = sample['context']['contexts']
        context_text = " ".join(contexts)[:1000]
        queries.append(f"{question} {context_text}")

    # Search
    print("Searching...")
    start = time.time()
    all_candidates = search(vectorizer, tfidf_matrix, metadata, queries, args.top_k)
    elapsed = time.time() - start
    print(f"Search completed in {elapsed:.1f}s ({elapsed/len(queries)*1000:.1f}ms/query)")

    # Build results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    for i, (sample, candidates) in enumerate(zip(dataset, all_candidates)):
        all_results.append({
            'query_idx': i,
            'pubid': sample['pubid'],
            'question': sample['question'],
            'final_decision': sample['final_decision'],
            'num_candidates': len(candidates),
            'top_score': candidates[0]['score'] if candidates else 0,
            'candidates': candidates
        })

    # Save
    output_file = OUTPUT_DIR / f"candidates_{args.subset}_top{args.top_k}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved to {output_file}")

    # Summary
    print("\n=== Summary ===")
    avg_score = sum(r['top_score'] for r in all_results) / len(all_results)
    print(f"Average top-1 score: {avg_score:.4f}")

    # Sample results
    print("\n=== Sample Results ===")
    for r in all_results[:3]:
        print(f"\nQ: {r['question'][:60]}...")
        if r['candidates']:
            c = r['candidates'][0]
            print(f"  Top-1 (score={c['score']:.4f}): {c['text_preview'][:80]}...")


if __name__ == "__main__":
    main()
