#!/usr/bin/env python3
"""
Download Dolma 3 health-related pre-training data from HuggingFace.

This script downloads the common_crawl-health-* subsets from the
allenai/dolma3_mix-6T-1025 dataset used to pre-train OLMo-3-7B.

Usage:
    conda activate pubmed-llm
    python scripts/download_dolma3_health.py

    # Download specific subsets only:
    python scripts/download_dolma3_health.py --subsets 0013 0014

    # List available subsets without downloading:
    python scripts/download_dolma3_health.py --list-only

    # Download to custom directory:
    python scripts/download_dolma3_health.py --output-dir /path/to/dir
"""

import argparse
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

# Configuration
DATASET_REPO = "allenai/dolma3_mix-6T-1025"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "dolma3_health"

# Known health subsets (as of Dec 2025)
HEALTH_SUBSETS = [
    "common_crawl-health-0013",
    "common_crawl-health-0014",
    "common_crawl-health-0015",
    "common_crawl-health-0016",
    "common_crawl-health-0017",
    "common_crawl-health-0018",
    "common_crawl-health-0020",
]


def list_health_subsets():
    """List all available health subsets in the dataset."""
    print(f"Scanning {DATASET_REPO} for health subsets...")
    api = HfApi()

    # List all files in the data directory
    files = api.list_repo_files(repo_id=DATASET_REPO, repo_type="dataset")

    # Find health-related folders
    health_folders = set()
    for f in files:
        if "health" in f.lower() and f.startswith("data/"):
            # Extract folder name (e.g., "data/common_crawl-health-0013/file.jsonl.zst")
            parts = f.split("/")
            if len(parts) >= 2:
                health_folders.add(parts[1])

    return sorted(health_folders)


def download_single_file(args):
    """Download a single file (for parallel execution)."""
    file_path, subset_dir, repo_id = args
    filename = Path(file_path).name
    local_path = subset_dir / filename

    if local_path.exists():
        return filename, "skipped"

    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=file_path,
        )
        shutil.copy2(downloaded_path, local_path)
        return filename, "downloaded"
    except Exception as e:
        return filename, f"error: {e}"


def download_subset(subset_name: str, output_dir: Path, repo_id: str = DATASET_REPO, workers: int = 8):
    """Download a specific health subset with parallel workers."""
    subset_dir = output_dir / subset_name
    subset_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {subset_name}...")

    api = HfApi()

    # List files in this subset
    all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    subset_files = [f for f in all_files if f.startswith(f"data/{subset_name}/")]

    if not subset_files:
        print(f"  Warning: No files found for {subset_name}")
        return 0

    print(f"  Found {len(subset_files)} files, downloading with {workers} workers...")

    # Prepare arguments for parallel download
    download_args = [(f, subset_dir, repo_id) for f in subset_files]

    downloaded = 0
    skipped = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_single_file, args): args for args in download_args}

        with tqdm(total=len(subset_files), desc=f"  {subset_name}", unit="file") as pbar:
            for future in as_completed(futures):
                filename, status = future.result()
                if status == "downloaded":
                    downloaded += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    errors += 1
                    tqdm.write(f"    Error: {filename} - {status}")
                pbar.update(1)

    print(f"  Done: {downloaded} downloaded, {skipped} skipped, {errors} errors")
    return downloaded + skipped


def main():
    parser = argparse.ArgumentParser(
        description="Download Dolma 3 health subsets for OLMo-3 pre-training data attribution"
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        help="Specific subset IDs to download (e.g., 0013 0014). Downloads all if not specified.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List available subsets without downloading",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel download workers (default: 8)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Dolma 3 Health Data Downloader")
    print(f"Dataset: {DATASET_REPO}")
    print("=" * 60)

    # List available subsets
    if args.list_only:
        subsets = list_health_subsets()
        print(f"\nAvailable health subsets ({len(subsets)}):")
        for s in subsets:
            print(f"  - {s}")
        return

    # Determine which subsets to download
    if args.subsets:
        # User specified subset IDs (e.g., "0013" -> "common_crawl-health-0013")
        subsets_to_download = [
            f"common_crawl-health-{sid}" if not sid.startswith("common_crawl") else sid
            for sid in args.subsets
        ]
    else:
        subsets_to_download = HEALTH_SUBSETS

    print(f"\nSubsets to download: {len(subsets_to_download)}")
    for s in subsets_to_download:
        print(f"  - {s}")

    print(f"\nOutput directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Download each subset
    total_files = 0
    for subset in subsets_to_download:
        count = download_subset(subset, args.output_dir, workers=args.workers)
        total_files += count

    print("\n" + "=" * 60)
    print(f"Download complete! Total files: {total_files}")
    print(f"Data saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
