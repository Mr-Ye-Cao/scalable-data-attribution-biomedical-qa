#!/usr/bin/env python3
"""
Step 1 of Controlled Attribution Experiment:
Strip QA template from fine-tuning data, keeping only raw content.

Input: data/rapidin_aligned/finetune_500_aligned.jsonl (templated)
Output: results/controlled_experiment/data/finetune_500_raw.jsonl (raw)

Template format being stripped:
- System prompt: "You are a clinical expert..."
- ChatML tags: <|im_start|>, <|im_end|>
- Prefixes: "Context:", "Final Decision:", "Long Answer:"
"""

import json
import re
from pathlib import Path

BASE_DIR = Path("/lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune")
INPUT_FILE = BASE_DIR / "data/rapidin_aligned/finetune_500_aligned.jsonl"
OUTPUT_DIR = BASE_DIR / "results/controlled_experiment/data"
OUTPUT_FILE = OUTPUT_DIR / "finetune_500_raw.jsonl"


def extract_context(instruction: str) -> str:
    """Extract raw context from templated instruction."""
    # Pattern: "Context: <content><|im_end|>"
    match = re.search(r'Context:\s*(.+?)<\|im_end\|>', instruction, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_long_answer(output: str) -> str:
    """Extract long answer from templated output."""
    # Pattern: "Long Answer: <content><|im_end|>"
    match = re.search(r'Long Answer:\s*(.+?)<\|im_end\|>', output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def strip_template(record: dict) -> dict:
    """
    Strip QA template from a record, keeping only raw content.

    Returns:
        dict with keys: pubid, context, question, final_decision, long_answer
    """
    return {
        "pubid": record["pubid"],
        "context": extract_context(record["instruction"]),
        "question": record["question"],
        "final_decision": record["final_decision"],
        "long_answer": extract_long_answer(record["output"]),
    }


def main():
    print("=" * 70)
    print("Step 1: Strip QA Template from Fine-tuning Data")
    print("=" * 70)

    print(f"\nInput: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process records
    raw_records = []
    with open(INPUT_FILE) as f:
        for line in f:
            record = json.loads(line)
            raw_record = strip_template(record)
            raw_records.append(raw_record)

    print(f"\nProcessed {len(raw_records)} records")

    # Write output
    with open(OUTPUT_FILE, 'w') as f:
        for record in raw_records:
            f.write(json.dumps(record) + '\n')

    print(f"Saved to: {OUTPUT_FILE}")

    # Show sample
    print("\n" + "=" * 70)
    print("SAMPLE OUTPUT (first record)")
    print("=" * 70)
    sample = raw_records[0]
    print(f"\nPubID: {sample['pubid']}")
    print(f"\nContext (first 300 chars):\n{sample['context'][:300]}...")
    print(f"\nQuestion:\n{sample['question']}")
    print(f"\nFinal Decision: {sample['final_decision']}")
    print(f"\nLong Answer (first 200 chars):\n{sample['long_answer'][:200]}...")

    # Verify no template remnants
    print("\n" + "=" * 70)
    print("VERIFICATION: Checking for template remnants")
    print("=" * 70)

    template_patterns = [
        "<|im_start|>",
        "<|im_end|>",
        "You are a clinical expert",
        "Context:",
        "Final Decision:",
        "Long Answer:",
    ]

    issues_found = False
    for record in raw_records:
        for field in ["context", "question", "long_answer"]:
            text = record[field]
            for pattern in template_patterns:
                if pattern in text:
                    print(f"WARNING: Found '{pattern}' in {field} of pubid {record['pubid']}")
                    issues_found = True

    if not issues_found:
        print("All records clean - no template remnants found!")

    print("\n" + "=" * 70)
    print("Step 1 Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
