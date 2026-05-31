#!/usr/bin/env python3
"""
Step 4 of Controlled Attribution Experiment:
Add QA template to pre-training samples to match fine-tuning data format.

This wraps pre-training documents in the same ChatML template used for fine-tuning,
making both datasets have identical structure for fair RapidIn comparison.

Strategy:
- Use document text as "context"
- Use highest-scoring relevant query as the "question"
- Generate synthetic answer (since pretrain docs aren't actual QA pairs)

Usage:
    python scripts/attribution/controlled_experiment/add_qa_template.py

Input: results/controlled_experiment/data/pretrain_500_raw.jsonl
Output: results/controlled_experiment/data/pretrain_500_aligned.jsonl
"""

import json
from pathlib import Path

BASE_DIR = Path("/lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune")
PRETRAIN_RAW = BASE_DIR / "results/controlled_experiment/data/pretrain_500_raw.jsonl"
FINETUNE_ALIGNED = BASE_DIR / "data/rapidin_aligned/finetune_500_aligned.jsonl"
OUTPUT_DIR = BASE_DIR / "results/controlled_experiment/data"

# ChatML template (same as fine-tuning)
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

# For pretrain docs, we use "maybe" since we don't know the actual answer
# The long answer summarizes that the context provides relevant information
OUTPUT_TEMPLATE = """Final Decision: maybe
Long Answer: The provided context contains relevant medical information that may help address this question. {summary}"""


def load_pretrain_raw() -> list[dict]:
    """Load raw pre-training subset."""
    records = []
    with open(PRETRAIN_RAW) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_finetune_sample() -> dict:
    """Load one fine-tuning sample to verify format."""
    with open(FINETUNE_ALIGNED) as f:
        return json.loads(f.readline())


def get_best_query(record: dict) -> dict:
    """Get the query with highest BM25 score for this document."""
    queries = record['relevant_queries']
    if not queries:
        return None
    # Sort by score (descending) and return best
    return max(queries, key=lambda q: q['score'])


def truncate_context(text: str, max_chars: int = 3000) -> str:
    """Truncate context to reasonable length (similar to fine-tuning data)."""
    if len(text) <= max_chars:
        return text
    # Try to truncate at sentence boundary
    truncated = text[:max_chars]
    last_period = truncated.rfind('.')
    if last_period > max_chars * 0.8:
        return truncated[:last_period + 1]
    return truncated + "..."


def extract_summary(text: str, max_chars: int = 200) -> str:
    """Extract a brief summary from the document for the long answer."""
    # Take first meaningful sentence(s) as summary
    sentences = text.split('.')
    summary = ""
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 20:  # Skip short fragments
            continue
        if len(summary) + len(sent) > max_chars:
            break
        summary += sent + ". "
    return summary.strip() if summary else "Further analysis of this literature is recommended."


def format_pretrain_record(record: dict) -> dict:
    """Format a pre-training record to match fine-tuning format."""
    # Get best relevant query
    best_query = get_best_query(record)
    if not best_query:
        return None

    # Truncate context (pretrain docs can be very long)
    context = truncate_context(record['text'])

    # Build instruction (same template as fine-tuning)
    instruction = INSTRUCTION_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        context=context,
        question=best_query['question']
    )

    # Build output (synthetic answer)
    summary = extract_summary(record['text'])
    output = OUTPUT_TEMPLATE.format(summary=summary)

    return {
        'instruction': instruction,
        'output': output,
        'doc_id': record['doc_id'],  # Use doc_id instead of pubid
        'question': best_query['question'],
        'final_decision': 'maybe',  # Synthetic decision
        'source_pubid': best_query['pubid'],  # The finetune query this is related to
        'bm25_score': best_query['score'],
        'frequency': record['frequency'],  # How many queries this doc is relevant to
    }


def main():
    print("=" * 70)
    print("Step 4: Add QA Template to Pre-training Samples")
    print("=" * 70)

    # Verify fine-tuning format
    print("\nVerifying fine-tuning data format...")
    ft_sample = load_finetune_sample()
    print(f"Fine-tuning keys: {list(ft_sample.keys())}")
    print(f"Instruction starts with: {ft_sample['instruction'][:50]}...")
    print(f"Output starts with: {ft_sample['output'][:50]}...")

    # Load pre-training data
    print(f"\nLoading pre-training data from {PRETRAIN_RAW}...")
    pretrain_records = load_pretrain_raw()
    print(f"Loaded {len(pretrain_records)} records")

    # Format records
    print("\nFormatting records with QA template...")
    formatted_records = []
    skipped = 0

    for record in pretrain_records:
        formatted = format_pretrain_record(record)
        if formatted:
            formatted_records.append(formatted)
        else:
            skipped += 1

    print(f"Formatted: {len(formatted_records)}, Skipped: {skipped}")

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "pretrain_500_aligned.jsonl"

    with open(output_file, 'w') as f:
        for record in formatted_records:
            f.write(json.dumps(record) + '\n')

    print(f"\nSaved to: {output_file}")

    # Verify alignment
    print("\n" + "=" * 70)
    print("FORMAT VERIFICATION")
    print("=" * 70)

    pt_sample = formatted_records[0]
    print("\n--- Fine-tuning Sample ---")
    print(f"Keys: {list(ft_sample.keys())}")
    print(f"Instruction length: {len(ft_sample['instruction'])} chars")
    print(f"Output length: {len(ft_sample['output'])} chars")

    print("\n--- Pre-training Sample (formatted) ---")
    print(f"Keys: {list(pt_sample.keys())}")
    print(f"Instruction length: {len(pt_sample['instruction'])} chars")
    print(f"Output length: {len(pt_sample['output'])} chars")

    # Check template structure matches
    ft_has_system = "<|im_start|>system" in ft_sample['instruction']
    pt_has_system = "<|im_start|>system" in pt_sample['instruction']
    ft_has_user = "<|im_start|>user" in ft_sample['instruction']
    pt_has_user = "<|im_start|>user" in pt_sample['instruction']
    ft_has_assistant = "<|im_start|>assistant" in ft_sample['instruction']
    pt_has_assistant = "<|im_start|>assistant" in pt_sample['instruction']

    print("\n--- Template Structure Check ---")
    print(f"Has <|im_start|>system: FT={ft_has_system}, PT={pt_has_system}")
    print(f"Has <|im_start|>user: FT={ft_has_user}, PT={pt_has_user}")
    print(f"Has <|im_start|>assistant: FT={ft_has_assistant}, PT={pt_has_assistant}")

    ft_starts_decision = ft_sample['output'].startswith("Final Decision:")
    pt_starts_decision = pt_sample['output'].startswith("Final Decision:")
    print(f"Output starts with 'Final Decision:': FT={ft_starts_decision}, PT={pt_starts_decision}")

    all_match = all([
        ft_has_system == pt_has_system,
        ft_has_user == pt_has_user,
        ft_has_assistant == pt_has_assistant,
        ft_starts_decision == pt_starts_decision,
    ])
    print(f"\n✓ All format checks passed!" if all_match else "\n✗ Format mismatch detected!")

    # Sample comparison
    print("\n" + "=" * 70)
    print("SAMPLE COMPARISON")
    print("=" * 70)

    print("\n--- Fine-tuning Instruction (first 400 chars) ---")
    print(ft_sample['instruction'][:400])

    print("\n--- Pre-training Instruction (first 400 chars) ---")
    print(pt_sample['instruction'][:400])

    print("\n--- Fine-tuning Output ---")
    print(ft_sample['output'][:200])

    print("\n--- Pre-training Output ---")
    print(pt_sample['output'][:200])

    print("\n" + "=" * 70)
    print("Step 4 Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
