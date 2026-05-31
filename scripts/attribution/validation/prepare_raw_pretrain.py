#!/usr/bin/env python3
"""
Convert pretrain_500_raw.jsonl to RapidIn format (instruction/input/output).
The raw data has no QA template - just plain text.
"""

import json
from pathlib import Path

BASE_DIR = Path("/lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune")

INPUT_FILE = BASE_DIR / "results/controlled_experiment/data/pretrain_500_raw.jsonl"
OUTPUT_FILE = BASE_DIR / "results/validation/data/pretrain_500_noformat.jsonl"

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with open(INPUT_FILE) as f:
        for line in f:
            data = json.loads(line)
            # Convert to RapidIn format: raw text as instruction, empty output
            record = {
                "instruction": data["text"],
                "input": "",
                "output": "",
                "doc_id": data["doc_id"]
            }
            records.append(record)

    with open(OUTPUT_FILE, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')

    print(f"Converted {len(records)} records")
    print(f"Output: {OUTPUT_FILE}")

    # Show sample
    print("\nSample (first 300 chars of instruction):")
    print(records[0]["instruction"][:300])

if __name__ == "__main__":
    main()
