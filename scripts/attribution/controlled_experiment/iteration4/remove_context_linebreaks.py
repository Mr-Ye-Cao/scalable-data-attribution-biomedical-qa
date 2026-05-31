#!/usr/bin/env python3
"""
Iteration 4: Remove line breaks from context part only

The pretrain data has many \n in the context (medical document text),
while finetune data has cleaner paragraph-style context.

This script removes \n from the context part while preserving
the prompt format line breaks.

Structure:
<|im_start|>system
You are a clinical expert...
Context:
[CONTEXT - remove \n here]<|im_end|>
<|im_start|>user
[QUESTION]<|im_end|>
<|im_start|>assistant
"""

import json
import re
from pathlib import Path

BASE_DIR = Path("/lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune")

INPUT_FILE = BASE_DIR / "results/controlled_experiment/iteration3/data/pretrain_500_full_match.jsonl"
OUTPUT_DIR = BASE_DIR / "results/controlled_experiment/iteration4/data"
OUTPUT_FILE = OUTPUT_DIR / "pretrain_500_no_linebreaks.jsonl"


def remove_context_linebreaks(instruction: str) -> str:
    """
    Remove \n from the context part only, preserving prompt format.

    The structure is:
    <|im_start|>system\n
    You are a clinical expert...\n
    Context: [CONTEXT WITH \n - CLEAN THIS]
    <|im_end|>\n
    <|im_start|>user\n
    [QUESTION]
    <|im_end|>\n
    <|im_start|>assistant
    """

    # Find the context section - look for "Context:" or "Context: "
    context_marker = instruction.find("Context:")
    if context_marker == -1:
        print("Warning: Could not find 'Context:' in instruction")
        return instruction

    # Find where context content starts (after "Context:" and optional space/newline)
    context_start = context_marker + len("Context:")
    # Skip any whitespace/newline immediately after "Context:"
    while context_start < len(instruction) and instruction[context_start] in ' \n':
        context_start += 1

    # Find where context ends (at first <|im_end|>)
    context_end = instruction.find("<|im_end|}", context_start)
    if context_end == -1:
        context_end = instruction.find("<|im_end|>", context_start)
    if context_end == -1:
        print("Warning: Could not find '<|im_end|>' after context")
        return instruction

    # Extract parts
    before_context = instruction[:context_start]
    context = instruction[context_start:context_end]
    after_context = instruction[context_end:]

    # Clean context: replace \n with space, collapse multiple spaces
    clean_context = context.replace("\n", " ")
    clean_context = re.sub(r'\s+', ' ', clean_context)
    clean_context = clean_context.strip()

    # Reconstruct
    return before_context + clean_context + after_context


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")

    records = []
    with open(INPUT_FILE) as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Loaded {len(records)} records")

    # Process each record
    processed = []
    for i, record in enumerate(records):
        new_record = record.copy()
        new_record['instruction'] = remove_context_linebreaks(record['instruction'])
        processed.append(new_record)

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        for record in processed:
            f.write(json.dumps(record) + '\n')

    print(f"Saved {len(processed)} records to {OUTPUT_FILE}")

    # Show example
    print("\n" + "="*60)
    print("EXAMPLE: First record")
    print("="*60)
    print("\n--- BEFORE (first 500 chars) ---")
    print(records[0]['instruction'][:500])
    print("\n--- AFTER (first 500 chars) ---")
    print(processed[0]['instruction'][:500])

    # Stats
    before_lens = [len(r['instruction']) for r in records]
    after_lens = [len(r['instruction']) for r in processed]

    print("\n" + "="*60)
    print("LENGTH STATS")
    print("="*60)
    print(f"Before: avg={sum(before_lens)/len(before_lens):.0f}, max={max(before_lens)}, min={min(before_lens)}")
    print(f"After:  avg={sum(after_lens)/len(after_lens):.0f}, max={max(after_lens)}, min={min(after_lens)}")


if __name__ == "__main__":
    main()
