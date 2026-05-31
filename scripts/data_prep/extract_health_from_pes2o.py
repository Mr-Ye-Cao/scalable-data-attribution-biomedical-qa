#!/usr/bin/env python3
"""
Extract health-related samples (Medicine, Biology) from pes2o data.
Keeps the complete JSON object for each matching sample.

Usage:
    python scripts/extract_health_from_pes2o.py <input.json.gz> <output.jsonl>

Example:
    python scripts/extract_health_from_pes2o.py data/olmo_mix_1124/pes2o-0025.json.gz data/olmo_mix_1124/pes2o-0025-health.jsonl
"""

import gzip
import json
import sys
from pathlib import Path

# Health-related fields to filter by
HEALTH_FIELDS = {'Medicine', 'Biology'}


def extract_health_samples(input_path: str, output_path: str):
    """Extract samples with Medicine or Biology in s2fieldsofstudy."""

    input_path = Path(input_path)
    output_path = Path(output_path)

    total_count = 0
    health_count = 0
    field_counts = {'Medicine': 0, 'Biology': 0, 'Both': 0}

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Filtering for: {HEALTH_FIELDS}")
    print()

    with gzip.open(input_path, 'rt', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            total_count += 1

            if total_count % 10000 == 0:
                print(f"Processed {total_count:,} samples, found {health_count:,} health samples...", flush=True)

            data = json.loads(line)
            fields = set(data.get('metadata', {}).get('s2fieldsofstudy', []))

            # Check if any health field is present
            matching_fields = fields & HEALTH_FIELDS

            if matching_fields:
                health_count += 1

                # Track field counts
                if 'Medicine' in matching_fields and 'Biology' in matching_fields:
                    field_counts['Both'] += 1
                elif 'Medicine' in matching_fields:
                    field_counts['Medicine'] += 1
                else:
                    field_counts['Biology'] += 1

                # Write complete JSON object
                f_out.write(json.dumps(data) + '\n')

    print()
    print("=" * 50)
    print(f"Total samples processed: {total_count:,}")
    print(f"Health samples extracted: {health_count:,} ({100*health_count/total_count:.1f}%)")
    print()
    print("Breakdown:")
    print(f"  Medicine only: {field_counts['Medicine']:,}")
    print(f"  Biology only:  {field_counts['Biology']:,}")
    print(f"  Both:          {field_counts['Both']:,}")
    print()
    print(f"Output saved to: {output_path}")

    return health_count


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    extract_health_samples(input_file, output_file)
