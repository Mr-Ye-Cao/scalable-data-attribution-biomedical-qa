#!/usr/bin/env python3
"""
Fix pretrain labels to match source query labels.

Instead of using "maybe" for all pretrain samples, use the actual
yes/no/maybe label from the source query (the finetune query this
pretrain doc was retrieved for).

This eliminates the label mismatch that caused pretrain influence
to be artificially low.
"""

import json
from pathlib import Path

BASE_DIR = Path("/lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune")

# Input files
PRETRAIN_ALIGNED = BASE_DIR / "results/controlled_experiment/data/pretrain_500_aligned.jsonl"
FINETUNE_ALIGNED = BASE_DIR / "data/rapidin_aligned/finetune_500_aligned.jsonl"

# Output
OUTPUT_FILE = BASE_DIR / "results/controlled_experiment/data/pretrain_500_label_matched.jsonl"

# Template for output
OUTPUT_TEMPLATE = """Final Decision: {decision}
Long Answer: {summary}"""


def load_finetune_labels() -> dict:
    """Load pubid -> final_decision mapping from finetune data."""
    labels = {}
    with open(FINETUNE_ALIGNED) as f:
        for line in f:
            record = json.loads(line)
            labels[record['pubid']] = record['final_decision']
    return labels


def extract_summary(text: str, max_chars: int = 200) -> str:
    """Extract summary from pretrain doc for long answer."""
    sentences = text.split('.')
    summary = ""
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 20:
            continue
        if len(summary) + len(sent) > max_chars:
            break
        summary += sent + ". "
    return summary.strip() if summary else "The context provides relevant medical information."


def main():
    print("=" * 70)
    print("Fix Pretrain Labels to Match Source Query Labels")
    print("=" * 70)

    # Load finetune labels
    print("\nLoading finetune labels...")
    finetune_labels = load_finetune_labels()
    print(f"Loaded {len(finetune_labels)} labels")

    # Count distribution
    label_counts = {}
    for label in finetune_labels.values():
        label_counts[label] = label_counts.get(label, 0) + 1
    print(f"Finetune label distribution: {label_counts}")

    # Process pretrain data
    print(f"\nProcessing pretrain data from {PRETRAIN_ALIGNED}...")

    fixed_records = []
    label_stats = {"yes": 0, "no": 0, "maybe": 0, "missing": 0}

    with open(PRETRAIN_ALIGNED) as f:
        for line in f:
            record = json.loads(line)
            source_pubid = str(record.get('source_pubid', ''))

            # Get label from source query
            if source_pubid in finetune_labels:
                new_label = finetune_labels[source_pubid]
                label_stats[new_label] += 1
            else:
                # Fallback to maybe if source not found
                new_label = "maybe"
                label_stats["missing"] += 1

            # Extract context from instruction for summary
            instruction = record['instruction']
            # Find context between "Context:" and "<|im_end|>"
            context_start = instruction.find("Context:") + len("Context:")
            context_end = instruction.find("<|im_end|>")
            if context_start > 0 and context_end > context_start:
                context = instruction[context_start:context_end].strip()
            else:
                context = instruction[:500]

            summary = extract_summary(context)

            # Create new output with matched label
            new_output = OUTPUT_TEMPLATE.format(
                decision=new_label,
                summary=summary
            )

            # Update record
            record['output'] = new_output
            record['final_decision'] = new_label
            record['original_decision'] = 'maybe'  # Keep track of original

            fixed_records.append(record)

    # Save output
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for record in fixed_records:
            f.write(json.dumps(record) + '\n')

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total records: {len(fixed_records)}")
    print(f"\nNew label distribution:")
    print(f"  yes:   {label_stats['yes']}")
    print(f"  no:    {label_stats['no']}")
    print(f"  maybe: {label_stats['maybe']}")
    if label_stats['missing'] > 0:
        print(f"  (missing source, defaulted to maybe): {label_stats['missing']}")

    # Verify
    print("\n" + "-" * 70)
    print("VERIFICATION - First 3 records:")
    print("-" * 70)
    for i, record in enumerate(fixed_records[:3]):
        print(f"\n[{i+1}] source_pubid: {record['source_pubid']}")
        print(f"    original: maybe -> new: {record['final_decision']}")
        print(f"    output preview: {record['output'][:80]}...")

    print("\n" + "=" * 70)
    print("Done! Now recompute pretrain gradients with label-matched data.")
    print("=" * 70)


if __name__ == "__main__":
    main()
