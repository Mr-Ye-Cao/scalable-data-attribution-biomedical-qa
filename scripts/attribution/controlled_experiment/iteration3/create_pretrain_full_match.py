#!/usr/bin/env python3
"""
Iteration 3: Create pretrain data with FULLY matched output.

Copy the ENTIRE output (Final Decision + Long Answer) from finetune data
to pretrain data. This eliminates both:
1. Decision mismatch (yes/no/maybe)
2. Long Answer content mismatch

Now the only difference is the Context (instruction) part.
"""

import json
from pathlib import Path

BASE_DIR = Path("/lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune")

# Input files
PRETRAIN_RAW = BASE_DIR / "results/controlled_experiment/data/pretrain_500_raw.jsonl"
FINETUNE_ALIGNED = BASE_DIR / "data/rapidin_aligned/finetune_500_aligned.jsonl"

# Output
OUTPUT_DIR = BASE_DIR / "results/controlled_experiment/iteration3/data"
OUTPUT_FILE = OUTPUT_DIR / "pretrain_500_full_match.jsonl"

# Template for instruction (same as finetune)
SYSTEM_PROMPT = """You are a clinical expert. Your task is to analyze the given medical literature context and then provide a Final Decision and a Long Answer."""

INSTRUCTION_TEMPLATE = """<|im_start|>system
{system_prompt}
Context: {context}
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
"""


def load_finetune_data() -> dict:
    """Load pubid -> full record mapping from finetune data."""
    data = {}
    with open(FINETUNE_ALIGNED) as f:
        for line in f:
            record = json.loads(line)
            data[record['pubid']] = record
    return data


def load_pretrain_raw() -> list[dict]:
    """Load raw pretrain data (with relevant_queries info)."""
    records = []
    with open(PRETRAIN_RAW) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def get_best_query(record: dict) -> dict:
    """Get the query with highest BM25 score for this document."""
    queries = record.get('relevant_queries', [])
    if not queries:
        return None
    return max(queries, key=lambda q: q['score'])


def truncate_context(text: str, max_chars: int = 3000) -> str:
    """Truncate context to reasonable length."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_period = truncated.rfind('.')
    if last_period > max_chars * 0.8:
        return truncated[:last_period + 1]
    return truncated + "..."


def main():
    print("=" * 70)
    print("Iteration 3: Create Pretrain with FULLY Matched Output")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load finetune data
    print("\nLoading finetune data...")
    finetune_data = load_finetune_data()
    print(f"Loaded {len(finetune_data)} finetune records")

    # Show finetune output example
    sample_ft = list(finetune_data.values())[0]
    print(f"\nFinetune output example:")
    print(f"  {sample_ft['output'][:100]}...")

    # Load pretrain data
    print(f"\nLoading pretrain data from {PRETRAIN_RAW}...")
    pretrain_records = load_pretrain_raw()
    print(f"Loaded {len(pretrain_records)} pretrain records")

    # Process pretrain data
    print("\nCreating fully-matched pretrain data...")

    output_records = []
    stats = {"matched": 0, "missing": 0}
    decision_dist = {"yes": 0, "no": 0, "maybe": 0}

    for record in pretrain_records:
        # Get best query for this pretrain doc
        best_query = get_best_query(record)
        if not best_query:
            stats["missing"] += 1
            continue

        source_pubid = str(best_query['pubid'])

        # Get finetune record for this source
        if source_pubid not in finetune_data:
            stats["missing"] += 1
            continue

        ft_record = finetune_data[source_pubid]
        stats["matched"] += 1

        # Build instruction with pretrain doc as context
        context = truncate_context(record['text'])
        instruction = INSTRUCTION_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            context=context,
            question=ft_record['question']  # Use finetune question
        )

        # Use ENTIRE finetune output (Final Decision + Long Answer)
        output = ft_record['output']
        decision = ft_record['final_decision']
        decision_dist[decision] += 1

        new_record = {
            'instruction': instruction,
            'output': output,  # FULLY copied from finetune
            'doc_id': record['doc_id'],
            'question': ft_record['question'],
            'final_decision': decision,
            'source_pubid': source_pubid,
            'bm25_score': best_query['score'],
        }
        output_records.append(new_record)

    # Save output
    with open(OUTPUT_FILE, 'w') as f:
        for record in output_records:
            f.write(json.dumps(record) + '\n')

    print(f"\nSaved to: {OUTPUT_FILE}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total records: {len(output_records)}")
    print(f"Matched: {stats['matched']}, Missing: {stats['missing']}")
    print(f"\nDecision distribution:")
    print(f"  yes:   {decision_dist['yes']}")
    print(f"  no:    {decision_dist['no']}")
    print(f"  maybe: {decision_dist['maybe']}")

    # Verification
    print("\n" + "=" * 70)
    print("VERIFICATION - Compare outputs")
    print("=" * 70)

    print("\n--- Finetune Output ---")
    print(sample_ft['output'][:300])

    print("\n--- Pretrain Output (should be IDENTICAL to some finetune) ---")
    print(output_records[0]['output'][:300])

    # Check if outputs match
    print("\n--- Match Check ---")
    pt_output = output_records[0]['output']
    source_pubid = output_records[0]['source_pubid']
    ft_output = finetune_data[source_pubid]['output']

    if pt_output == ft_output:
        print("✓ Pretrain output EXACTLY matches finetune output!")
    else:
        print("✗ Mismatch detected")
        print(f"  PT: {pt_output[:100]}")
        print(f"  FT: {ft_output[:100]}")

    print("\n" + "=" * 70)
    print("Done! Now run gradient computation.")
    print("=" * 70)


if __name__ == "__main__":
    main()
