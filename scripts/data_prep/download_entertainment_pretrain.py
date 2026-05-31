#!/usr/bin/env python3
"""
Download entertainment (non-medical) pre-training data as a control group.

This serves as a negative control to validate that RapidIn correctly identifies
that medical pretrain data has higher influence than unrelated entertainment data
on medical QA tasks.

Source: allenai/dolma3_mix-6T-1025-7B (OLMo-3 pretraining data)
Subset: common_crawl-entertainment-0014

Usage:
    python scripts/data_prep/download_entertainment_pretrain.py
"""

import json
import zstandard as zstd
from pathlib import Path
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# Configuration
DATASET_REPO = "allenai/dolma3_mix-6T-1025-7B"
FILE_PATH = "data/common_crawl-entertainment-0014/shard_00000469.jsonl.zst"
NUM_SAMPLES = 500

BASE_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = BASE_DIR / "entertainment-pretrain-data"


def download_and_extract():
    """Download and extract entertainment pretrain data."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Entertainment Pretrain Data Downloader (Control Group)")
    print(f"Dataset: {DATASET_REPO}")
    print(f"File: {FILE_PATH}")
    print(f"Target samples: {NUM_SAMPLES}")
    print("=" * 60)

    # Download the file
    print("\n[1/3] Downloading from HuggingFace...")
    local_path = hf_hub_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        filename=FILE_PATH,
    )
    print(f"Downloaded to: {local_path}")

    # Decompress and extract samples
    print(f"\n[2/3] Extracting {NUM_SAMPLES} samples...")
    samples = []

    dctx = zstd.ZstdDecompressor()
    with open(local_path, 'rb') as compressed:
        with dctx.stream_reader(compressed) as reader:
            text_stream = reader.read().decode('utf-8')
            lines = text_stream.strip().split('\n')

            for i, line in enumerate(tqdm(lines[:NUM_SAMPLES], desc="Extracting")):
                if line.strip():
                    try:
                        data = json.loads(line)
                        samples.append({
                            "id": f"entertainment_{i:04d}",
                            "text": data.get("text", ""),
                            "source": data.get("source", "common_crawl-entertainment"),
                            "metadata": data.get("metadata", {})
                        })
                    except json.JSONDecodeError:
                        continue

                if len(samples) >= NUM_SAMPLES:
                    break

    print(f"Extracted {len(samples)} samples")

    # Save raw samples
    print("\n[3/3] Saving data...")
    raw_output = OUTPUT_DIR / "entertainment_500_raw.jsonl"
    with open(raw_output, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    print(f"Saved raw data to: {raw_output}")

    # Show sample
    print("\n" + "=" * 60)
    print("Sample entry:")
    print("=" * 60)
    if samples:
        sample = samples[0]
        print(f"ID: {sample['id']}")
        print(f"Source: {sample['source']}")
        print(f"Text (first 500 chars): {sample['text'][:500]}...")

    print("\n" + "=" * 60)
    print(f"Download complete! {len(samples)} samples saved to:")
    print(f"  {raw_output}")
    print("=" * 60)

    return samples


if __name__ == "__main__":
    download_and_extract()
