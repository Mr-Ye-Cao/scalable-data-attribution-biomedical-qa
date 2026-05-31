#!/usr/bin/env python3
"""
Extract embeddings from dolma3_health data using vLLM pooling.

This script uses vLLM's runner='pooling' mode to extract last-token embeddings
at ~210 samples/sec (~4.76ms per sample), which is 4.6x faster than HuggingFace.

Usage:
    conda activate pubmed-gh200

    # Process all data
    python scripts/extract_embeddings_vllm.py

    # Process specific subset
    python scripts/extract_embeddings_vllm.py --subset common_crawl-health-0013

    # Test with small sample
    python scripts/extract_embeddings_vllm.py --max_samples 1000

Output:
    data/embeddings/
    ├── embeddings_0000.npy      # Embedding chunks [N, 4096] float16
    ├── embeddings_0001.npy
    ├── ...
    ├── metadata.jsonl           # Sample IDs and text previews
    └── progress.json            # For resume capability
"""

import argparse
import json
import time
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import zstandard as zstd
from tqdm import tqdm
from vllm import LLM


# Configuration
MODEL_PATH = "./OLMo-3-1025-7B"
DATA_DIR = Path("./data/dolma3_health")
OUTPUT_DIR = Path("./data/embeddings")
MAX_TOKENS = 500  # Truncate to this many tokens
CHUNK_SIZE = 50000  # Save embeddings in chunks
BATCH_SIZE = 1000  # Process this many at once (vLLM handles internal batching)


def load_model():
    """Load OLMo-3-7B with vLLM pooling runner."""
    print(f"Loading model from {MODEL_PATH} with vLLM pooling...")
    print("This may take ~15-20 seconds for graph compilation...")

    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        trust_remote_code=True,
        runner="pooling",  # Convert to embedding model
        max_model_len=512,
        gpu_memory_utilization=0.9,
    )

    print("Model loaded successfully!")
    return llm


def read_zst_file(filepath):
    """Read a zstandard compressed JSONL file."""
    dctx = zstd.ZstdDecompressor()
    records = []

    with open(filepath, 'rb') as f:
        with dctx.stream_reader(f) as reader:
            content = reader.read().decode('utf-8')
            for line in content.strip().split('\n'):
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    continue

    return records


def get_all_shards(data_dir):
    """Get all zst shard files from all subsets."""
    shards = []
    for subset_dir in sorted(data_dir.iterdir()):
        if subset_dir.is_dir():
            for shard_file in sorted(subset_dir.glob("*.jsonl.zst")):
                shards.append(shard_file)
    return shards


def save_progress(progress_file, processed_shards, total_samples):
    """Save progress for resume capability."""
    with open(progress_file, 'w') as f:
        json.dump({
            "processed_shards": list(processed_shards),
            "total_samples": total_samples,
            "timestamp": datetime.now().isoformat(),
        }, f)


def load_progress(progress_file):
    """Load progress from checkpoint."""
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            data = json.load(f)
            return set(data["processed_shards"]), data["total_samples"]
    return set(), 0


def extract_embeddings_batch(llm, texts):
    """Extract embeddings for a batch of texts using vLLM."""
    outputs = llm.embed(texts, truncate_prompt_tokens=MAX_TOKENS, use_tqdm=False)
    embeddings = np.array([o.outputs.embedding for o in outputs], dtype=np.float32)
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings using vLLM")
    parser.add_argument("--subset", type=str, help="Process only this subset")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--max_samples", type=int, help="Max samples to process (for testing)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for processing")
    args = parser.parse_args()

    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress_file = OUTPUT_DIR / "progress.json"
    metadata_file = OUTPUT_DIR / "metadata.jsonl"

    # Load progress if resuming
    if args.resume:
        processed_shards, total_samples = load_progress(progress_file)
        print(f"Resuming: {len(processed_shards)} shards processed, {total_samples} samples")
    else:
        processed_shards = set()
        total_samples = 0
        # Clear metadata file
        if metadata_file.exists():
            metadata_file.unlink()

    # Get shards to process
    all_shards = get_all_shards(DATA_DIR)
    if args.subset:
        all_shards = [s for s in all_shards if args.subset in str(s)]

    shards_to_process = [s for s in all_shards if str(s) not in processed_shards]

    print(f"\n{'='*60}")
    print("Dolma3 Health Embedding Extraction (vLLM)")
    print(f"{'='*60}")
    print(f"Total shards: {len(all_shards)}")
    print(f"Already processed: {len(processed_shards)}")
    print(f"To process: {len(shards_to_process)}")
    print(f"Max tokens per sample: {MAX_TOKENS}")
    print(f"Batch size: {args.batch_size}")

    if not shards_to_process:
        print("Nothing to process!")
        return

    # Load model
    llm = load_model()

    # Track embeddings for current chunk
    chunk_embeddings = []
    chunk_id = total_samples // CHUNK_SIZE

    # Process shards
    start_time = time.time()
    samples_this_run = 0

    for shard_path in tqdm(shards_to_process, desc="Shards"):
        try:
            records = read_zst_file(shard_path)
        except Exception as e:
            print(f"\nError reading {shard_path}: {e}")
            continue

        # Extract texts
        texts = []
        valid_records = []
        for r in records:
            text = r.get('text', '')
            if text:
                texts.append(text[:2000])  # Pre-truncate chars
                valid_records.append(r)

        if not texts:
            processed_shards.add(str(shard_path))
            continue

        # Process in batches
        for i in range(0, len(texts), args.batch_size):
            batch_texts = texts[i:i+args.batch_size]
            batch_records = valid_records[i:i+args.batch_size]

            try:
                embeddings = extract_embeddings_batch(llm, batch_texts)
            except Exception as e:
                print(f"\nError processing batch: {e}")
                continue

            # Save metadata
            with open(metadata_file, 'a') as f:
                for j, record in enumerate(batch_records):
                    meta = {
                        "idx": total_samples + j,
                        "id": record.get("id", ""),
                        "text_preview": record.get("text", "")[:200],
                        "shard": str(shard_path),
                    }
                    f.write(json.dumps(meta) + "\n")

            # Accumulate embeddings
            chunk_embeddings.append(embeddings)
            total_samples += len(embeddings)
            samples_this_run += len(embeddings)

            # Save chunk if full
            if total_samples >= (chunk_id + 1) * CHUNK_SIZE:
                chunk_data = np.vstack(chunk_embeddings)
                chunk_file = OUTPUT_DIR / f"embeddings_{chunk_id:04d}.npy"
                np.save(chunk_file, chunk_data.astype(np.float16))
                print(f"\nSaved chunk {chunk_id}: {chunk_data.shape}")
                chunk_embeddings = []
                chunk_id += 1

            # Check max samples limit
            if args.max_samples and total_samples >= args.max_samples:
                print(f"\nReached max_samples limit: {args.max_samples}")
                break

        # Mark shard as processed
        processed_shards.add(str(shard_path))
        save_progress(progress_file, processed_shards, total_samples)

        if args.max_samples and total_samples >= args.max_samples:
            break

    # Save remaining embeddings
    if chunk_embeddings:
        chunk_data = np.vstack(chunk_embeddings)
        chunk_file = OUTPUT_DIR / f"embeddings_{chunk_id:04d}.npy"
        np.save(chunk_file, chunk_data.astype(np.float16))
        print(f"\nSaved final chunk {chunk_id}: {chunk_data.shape}")

    # Summary
    elapsed = time.time() - start_time
    throughput = samples_this_run / elapsed if elapsed > 0 else 0

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"Total samples processed: {total_samples:,}")
    print(f"This run: {samples_this_run:,} samples in {elapsed:.1f}s")
    print(f"Throughput: {throughput:.1f} samples/sec")
    print(f"Output directory: {OUTPUT_DIR}")

    # List output files
    print(f"\nOutput files:")
    total_size = 0
    for f in sorted(OUTPUT_DIR.glob("embeddings_*.npy")):
        size = f.stat().st_size / 1e9
        total_size += size
        print(f"  {f.name}: {size:.2f} GB")
    print(f"  Total: {total_size:.2f} GB")


if __name__ == "__main__":
    main()
