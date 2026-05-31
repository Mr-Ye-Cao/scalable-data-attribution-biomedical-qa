#!/usr/bin/env python3
"""
Download PubMedQA dataset from HuggingFace.

Dataset: qiaojin/PubMedQA
Source: https://huggingface.co/datasets/qiaojin/PubMedQA

Subsets:
- pqa_labeled: 1,000 examples with expert labels (yes/no/maybe)
- pqa_unlabeled: 61,249 examples without final_decision labels
- pqa_artificial: 211,269 examples with machine-generated labels

Usage:
    conda activate pubmed-llm
    python scripts/download_pubmedqa.py
"""

import os
from pathlib import Path
from datasets import load_dataset

# Configuration
DATASET_NAME = "qiaojin/PubMedQA"
SUBSETS = ["pqa_labeled", "pqa_unlabeled", "pqa_artificial"]
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "pubmedqa"


def download_pubmedqa():
    """Download all PubMedQA subsets and save to disk."""

    print(f"Downloading PubMedQA from: {DATASET_NAME}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 50)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for subset in SUBSETS:
        print(f"\nDownloading subset: {subset}")

        # Load dataset from HuggingFace
        dataset = load_dataset(DATASET_NAME, subset)

        # Print info
        print(f"  Splits: {list(dataset.keys())}")
        for split_name, split_data in dataset.items():
            print(f"  {split_name}: {len(split_data)} examples")
            print(f"  Features: {list(split_data.features.keys())}")

        # Save to disk
        subset_dir = OUTPUT_DIR / subset
        dataset.save_to_disk(subset_dir)
        print(f"  Saved to: {subset_dir}")

    print("\n" + "=" * 50)
    print("Download complete!")
    print(f"Data saved to: {OUTPUT_DIR}")

    # Print summary
    print("\nSummary:")
    print("  - pqa_labeled:    1,000 examples (expert-labeled, use for fine-tuning)")
    print("  - pqa_unlabeled: 61,249 examples (no labels)")
    print("  - pqa_artificial: 211,269 examples (machine-generated labels)")


if __name__ == "__main__":
    download_pubmedqa()
