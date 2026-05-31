#!/usr/bin/env python3
"""
Process all 26 pes2o shards: download → extract health samples → delete original.

This script processes each shard one at a time to minimize disk usage.
Health samples (Medicine, Biology) are extracted and saved to one file per shard.

Usage:
    python scripts/process_all_pes2o_health.py

Output:
    data/olmo_mix_1124/pes2o-0000-health.jsonl
    data/olmo_mix_1124/pes2o-0001-health.jsonl
    ...
    data/olmo_mix_1124/pes2o-0025-health.jsonl
    data/olmo_mix_1124/pes2o-health-log.txt  (field counts for all shards)
"""

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

# Configuration
BASE_URL = "https://huggingface.co/datasets/allenai/olmo-mix-1124/resolve/main/data/pes2o"
NUM_SHARDS = 26  # pes2o-0000.json.gz to pes2o-0025.json.gz
HEALTH_FIELDS = {'Medicine', 'Biology'}

# Paths
DATA_DIR = Path("data/olmo_mix_1124")
PROGRESS_FILE = DATA_DIR / "pes2o-health-progress.json"
LOG_FILE = DATA_DIR / "pes2o-health-log.txt"


def get_output_path(shard_id: int) -> Path:
    """Get output path for a shard."""
    return DATA_DIR / f"pes2o-{shard_id:04d}-health.jsonl"


def download_shard(shard_id: int) -> Path:
    """Download a single shard and return its path."""
    filename = f"pes2o-{shard_id:04d}.json.gz"
    url = f"{BASE_URL}/{filename}"
    output_path = DATA_DIR / filename

    print(f"  Downloading {filename}...")
    result = subprocess.run(
        ["wget", "-q", "--show-progress", url, "-O", str(output_path)],
        capture_output=False
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to download {filename}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Downloaded: {size_mb:.1f} MB")

    return output_path


def count_fields_and_extract(input_path: Path, output_path: Path, log_file) -> tuple:
    """
    Count all s2fieldsofstudy fields, log them, then extract health samples.
    Returns (total_count, health_count, field_counts).
    """
    health_count = 0
    total_count = 0
    field_counts = Counter()

    with gzip.open(input_path, 'rt', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            total_count += 1
            data = json.loads(line)
            fields = data.get('metadata', {}).get('s2fieldsofstudy', [])

            # Count all fields
            for f in fields:
                field_counts[f] += 1

            # Extract health samples
            if set(fields) & HEALTH_FIELDS:
                health_count += 1
                f_out.write(json.dumps(data) + '\n')

    return total_count, health_count, field_counts


def write_shard_log(log_file, shard_id: int, total_count: int, health_count: int, field_counts: Counter):
    """Write field counts for a shard to the log file."""
    log_file.write(f"\n{'='*60}\n")
    log_file.write(f"Shard: pes2o-{shard_id:04d}.json.gz\n")
    log_file.write(f"Timestamp: {datetime.now().isoformat()}\n")
    log_file.write(f"{'='*60}\n")
    log_file.write(f"Total samples: {total_count:,}\n")
    log_file.write(f"Health samples (Medicine + Biology): {health_count:,}\n")
    log_file.write(f"\nField counts (sorted by count):\n")
    log_file.write(f"{'-'*40}\n")
    for field, count in sorted(field_counts.items(), key=lambda x: -x[1]):
        log_file.write(f"{field}: {count:,}\n")
    log_file.flush()


def save_progress(progress: dict):
    """Save progress to file for resume capability."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def load_progress() -> dict:
    """Load progress from file if exists."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {
        'completed_shards': [],
        'total_samples': 0,
        'health_samples': 0,
        'started_at': datetime.now().isoformat()
    }


def main():
    # Create data directory if needed
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load progress (for resume capability)
    progress = load_progress()
    completed = set(progress['completed_shards'])

    print("=" * 60)
    print("Processing all pes2o shards for health-related content")
    print("=" * 60)
    print(f"Output directory: {DATA_DIR}")
    print(f"Output format: pes2o-XXXX-health.jsonl (one per shard)")
    print(f"Log file: {LOG_FILE}")
    print(f"Health fields: {HEALTH_FIELDS}")
    print(f"Already completed: {len(completed)}/{NUM_SHARDS} shards")
    print()

    # Open log file in append mode
    log_mode = 'a' if completed else 'w'
    with open(LOG_FILE, log_mode, encoding='utf-8') as log_file:
        # Write header if new log
        if log_mode == 'w':
            log_file.write("PES2O Health Data Extraction Log\n")
            log_file.write(f"Started: {datetime.now().isoformat()}\n")
            log_file.write(f"Health fields: {HEALTH_FIELDS}\n")

        for shard_id in range(NUM_SHARDS):
            if shard_id in completed:
                print(f"[{shard_id+1:02d}/{NUM_SHARDS}] Shard {shard_id:04d} - already processed, skipping")
                continue

            print(f"\n[{shard_id+1:02d}/{NUM_SHARDS}] Processing shard {shard_id:04d}")
            print("-" * 40)

            try:
                # Step 1: Download
                shard_path = download_shard(shard_id)

                # Step 2: Count fields and extract health samples
                output_path = get_output_path(shard_id)
                print(f"  Counting fields and extracting health samples...")
                total_count, health_count, field_counts = count_fields_and_extract(
                    shard_path, output_path, log_file
                )
                print(f"  Total samples: {total_count:,}")
                print(f"  Health samples: {health_count:,} ({100*health_count/total_count:.1f}%)")
                print(f"  Saved to: {output_path.name}")

                # Step 3: Write to log file
                print(f"  Writing to log file...")
                write_shard_log(log_file, shard_id, total_count, health_count, field_counts)

                # Step 4: Delete original
                print(f"  Deleting original file...")
                shard_path.unlink()
                print(f"  Deleted: {shard_path.name}")

                # Update progress
                progress['completed_shards'].append(shard_id)
                progress['total_samples'] += total_count
                progress['health_samples'] += health_count
                save_progress(progress)

            except Exception as e:
                print(f"  ERROR: {e}")
                print(f"  Saving progress and exiting...")
                save_progress(progress)
                sys.exit(1)

        # Write final summary to log
        log_file.write(f"\n{'='*60}\n")
        log_file.write(f"FINAL SUMMARY\n")
        log_file.write(f"{'='*60}\n")
        log_file.write(f"Completed: {datetime.now().isoformat()}\n")
        log_file.write(f"Total shards: {NUM_SHARDS}\n")
        log_file.write(f"Total samples: {progress['total_samples']:,}\n")
        log_file.write(f"Health samples: {progress['health_samples']:,}\n")
        log_file.write(f"Health percentage: {100*progress['health_samples']/progress['total_samples']:.1f}%\n")

    # Final summary to console
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"Total shards processed: {NUM_SHARDS}")
    print(f"Total samples scanned:  {progress['total_samples']:,}")
    print(f"Health samples extracted: {progress['health_samples']:,} ({100*progress['health_samples']/progress['total_samples']:.1f}%)")
    print(f"Output files: {DATA_DIR}/pes2o-XXXX-health.jsonl")
    print(f"Log file: {LOG_FILE}")

    # Calculate total output size
    total_size = sum(get_output_path(i).stat().st_size for i in range(NUM_SHARDS) if get_output_path(i).exists())
    print(f"Total output size: {total_size / (1024*1024*1024):.2f} GB")

    # Clean up progress file
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        print(f"Cleaned up progress file")


if __name__ == '__main__':
    main()
