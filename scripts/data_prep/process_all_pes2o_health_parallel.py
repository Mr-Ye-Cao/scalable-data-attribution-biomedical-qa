#!/usr/bin/env python3
"""
Process all 26 pes2o shards with PARALLEL downloads.

Downloads multiple shards concurrently to maximize network bandwidth,
then processes each as it completes.

Usage:
    python scripts/process_all_pes2o_health_parallel.py [--workers N]

Output:
    data/olmo_mix_1124/pes2o-0000-health.jsonl
    ...
    data/olmo_mix_1124/pes2o-0025-health.jsonl
    data/olmo_mix_1124/pes2o-health-log.txt
"""

import gzip
import json
import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configuration
BASE_URL = "https://huggingface.co/datasets/allenai/olmo-mix-1124/resolve/main/data/pes2o"
NUM_SHARDS = 26
HEALTH_FIELDS = {'Medicine', 'Biology'}
DEFAULT_WORKERS = 4  # Number of parallel downloads

# Paths
DATA_DIR = Path("data/olmo_mix_1124")
PROGRESS_FILE = DATA_DIR / "pes2o-health-progress.json"
LOG_FILE = DATA_DIR / "pes2o-health-log.txt"

# Thread lock for log file writing
log_lock = threading.Lock()
progress_lock = threading.Lock()


def get_output_path(shard_id: int) -> Path:
    return DATA_DIR / f"pes2o-{shard_id:04d}-health.jsonl"


def get_shard_path(shard_id: int) -> Path:
    return DATA_DIR / f"pes2o-{shard_id:04d}.json.gz"


def download_shard(shard_id: int) -> Path:
    """Download a single shard and return its path."""
    filename = f"pes2o-{shard_id:04d}.json.gz"
    url = f"{BASE_URL}/{filename}"
    output_path = DATA_DIR / filename

    print(f"  [{shard_id:04d}] Downloading...")
    result = subprocess.run(
        ["wget", "-q", url, "-O", str(output_path)],
        capture_output=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to download {filename}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  [{shard_id:04d}] Downloaded: {size_mb:.1f} MB")

    return output_path


def count_fields_and_extract(input_path: Path, output_path: Path) -> tuple:
    """Count all fields and extract health samples."""
    health_count = 0
    total_count = 0
    field_counts = Counter()

    with gzip.open(input_path, 'rt', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            total_count += 1
            data = json.loads(line)
            fields = data.get('metadata', {}).get('s2fieldsofstudy', [])

            for f in fields:
                field_counts[f] += 1

            if set(fields) & HEALTH_FIELDS:
                health_count += 1
                f_out.write(json.dumps(data) + '\n')

    return total_count, health_count, field_counts


def write_shard_log(log_file, shard_id: int, total_count: int, health_count: int, field_counts: Counter):
    """Write field counts for a shard to the log file (thread-safe)."""
    log_entry = []
    log_entry.append(f"\n{'='*60}")
    log_entry.append(f"Shard: pes2o-{shard_id:04d}.json.gz")
    log_entry.append(f"Timestamp: {datetime.now().isoformat()}")
    log_entry.append(f"{'='*60}")
    log_entry.append(f"Total samples: {total_count:,}")
    log_entry.append(f"Health samples (Medicine + Biology): {health_count:,}")
    log_entry.append(f"\nField counts (sorted by count):")
    log_entry.append(f"{'-'*40}")
    for field, count in sorted(field_counts.items(), key=lambda x: -x[1]):
        log_entry.append(f"{field}: {count:,}")

    with log_lock:
        log_file.write('\n'.join(log_entry) + '\n')
        log_file.flush()


def save_progress(progress: dict):
    """Save progress to file (thread-safe)."""
    with progress_lock:
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


def process_shard(shard_id: int, log_file, progress: dict) -> tuple:
    """Download, process, and delete a single shard. Returns (shard_id, total, health)."""
    try:
        # Step 1: Download
        shard_path = download_shard(shard_id)

        # Step 2: Count fields and extract
        output_path = get_output_path(shard_id)
        print(f"  [{shard_id:04d}] Extracting health samples...")
        total_count, health_count, field_counts = count_fields_and_extract(shard_path, output_path)
        print(f"  [{shard_id:04d}] Extracted: {health_count:,} / {total_count:,} ({100*health_count/total_count:.1f}%)")

        # Step 3: Write to log
        write_shard_log(log_file, shard_id, total_count, health_count, field_counts)

        # Step 4: Delete original
        shard_path.unlink()
        print(f"  [{shard_id:04d}] DONE - deleted original")

        # Update progress
        with progress_lock:
            progress['completed_shards'].append(shard_id)
            progress['total_samples'] += total_count
            progress['health_samples'] += health_count
        save_progress(progress)

        return shard_id, total_count, health_count

    except Exception as e:
        print(f"  [{shard_id:04d}] ERROR: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Process pes2o shards in parallel')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                        help=f'Number of parallel downloads (default: {DEFAULT_WORKERS})')
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    progress = load_progress()
    completed = set(progress['completed_shards'])
    remaining = [i for i in range(NUM_SHARDS) if i not in completed]

    print("=" * 60)
    print("Processing pes2o shards (PARALLEL)")
    print("=" * 60)
    print(f"Parallel workers: {args.workers}")
    print(f"Output directory: {DATA_DIR}")
    print(f"Log file: {LOG_FILE}")
    print(f"Health fields: {HEALTH_FIELDS}")
    print(f"Already completed: {len(completed)}/{NUM_SHARDS} shards")
    print(f"Remaining: {len(remaining)} shards")
    print()

    if not remaining:
        print("All shards already processed!")
        return

    start_time = datetime.now()

    log_mode = 'a' if completed else 'w'
    with open(LOG_FILE, log_mode, encoding='utf-8') as log_file:
        if log_mode == 'w':
            log_file.write("PES2O Health Data Extraction Log (Parallel)\n")
            log_file.write(f"Started: {datetime.now().isoformat()}\n")
            log_file.write(f"Workers: {args.workers}\n")
            log_file.write(f"Health fields: {HEALTH_FIELDS}\n")

        # Process shards in parallel
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_shard, shard_id, log_file, progress): shard_id
                for shard_id in remaining
            }

            completed_count = len(completed)
            for future in as_completed(futures):
                shard_id = futures[future]
                try:
                    _, total, health = future.result()
                    completed_count += 1
                    print(f"\n[{completed_count}/{NUM_SHARDS}] Shard {shard_id:04d} completed")
                except Exception as e:
                    print(f"\n[ERROR] Shard {shard_id:04d} failed: {e}")

        # Write final summary
        log_file.write(f"\n{'='*60}\n")
        log_file.write(f"FINAL SUMMARY\n")
        log_file.write(f"{'='*60}\n")
        log_file.write(f"Completed: {datetime.now().isoformat()}\n")
        log_file.write(f"Total shards: {NUM_SHARDS}\n")
        log_file.write(f"Total samples: {progress['total_samples']:,}\n")
        log_file.write(f"Health samples: {progress['health_samples']:,}\n")
        if progress['total_samples'] > 0:
            log_file.write(f"Health percentage: {100*progress['health_samples']/progress['total_samples']:.1f}%\n")

    elapsed = datetime.now() - start_time

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"Time elapsed: {elapsed}")
    print(f"Total shards: {NUM_SHARDS}")
    print(f"Total samples: {progress['total_samples']:,}")
    if progress['total_samples'] > 0:
        print(f"Health samples: {progress['health_samples']:,} ({100*progress['health_samples']/progress['total_samples']:.1f}%)")
    print(f"Output files: {DATA_DIR}/pes2o-XXXX-health.jsonl")
    print(f"Log file: {LOG_FILE}")

    total_size = sum(get_output_path(i).stat().st_size for i in range(NUM_SHARDS) if get_output_path(i).exists())
    print(f"Total output size: {total_size / (1024*1024*1024):.2f} GB")

    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()


if __name__ == '__main__':
    main()
