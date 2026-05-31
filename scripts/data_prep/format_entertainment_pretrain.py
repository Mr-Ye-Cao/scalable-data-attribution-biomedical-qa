#!/usr/bin/env python3
"""
Format entertainment pretrain data with QA template (like Iter1).

This creates a formatted version that matches the medical pretrain format,
allowing fair comparison of influence scores.

Format: ChatML template with entertainment text as context + test questions
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
ENTERTAINMENT_RAW = BASE_DIR / "entertainment-pretrain-data" / "entertainment_500_raw.jsonl"
TEST_DATA = BASE_DIR / "data-aggregate/input/test/test_500.jsonl"
OUTPUT_DIR = BASE_DIR / "entertainment-pretrain-data"


def load_jsonl(path):
    """Load JSONL file."""
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def create_qa_template(context: str, question: str, answer: str = "maybe") -> str:
    """Create ChatML formatted instruction."""
    # Truncate context if too long (match medical pretrain behavior)
    max_context_len = 3000
    if len(context) > max_context_len:
        context = context[:max_context_len] + "..."

    template = f"""<|im_start|>system
You are a clinical expert. Your task is to analyze the given medical literature context and then provide a Final Decision and a Long Answer.
Context: {context}
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
"""
    return template


def create_output(context: str, answer: str = "maybe") -> str:
    """Create output in the same format as Iter1."""
    # Take first ~200 chars of context for long answer
    snippet = context[:200].replace('\n', ' ')
    return f"Final Decision: {answer}\nLong Answer: The provided context contains relevant information. {snippet}"


def main():
    print("=" * 60)
    print("Formatting Entertainment Pretrain Data (QA Template)")
    print("=" * 60)

    # Load data
    print("\n[1/3] Loading data...")
    entertainment_data = load_jsonl(ENTERTAINMENT_RAW)
    test_data = load_jsonl(TEST_DATA)

    print(f"  Entertainment samples: {len(entertainment_data)}")
    print(f"  Test questions: {len(test_data)}")

    # Format each sample
    print("\n[2/3] Formatting samples...")
    formatted_samples = []

    for i, ent_sample in enumerate(entertainment_data):
        # Pair with corresponding test question (cycling if needed)
        test_sample = test_data[i % len(test_data)]

        context = ent_sample["text"]
        question = test_sample["question"]

        instruction = create_qa_template(context, question)
        output = create_output(context)

        formatted = {
            "instruction": instruction,
            "input": "",
            "output": output,
            "doc_id": ent_sample["id"],
            "question": question,
            "final_decision": "maybe",
            "source": "entertainment",
            "original_source": ent_sample.get("source", "common_crawl-entertainment")
        }
        formatted_samples.append(formatted)

    # Save formatted data
    print("\n[3/3] Saving formatted data...")

    # Save formatted version (like iter1)
    formatted_output = OUTPUT_DIR / "entertainment_500_formatted.jsonl"
    with open(formatted_output, 'w') as f:
        for sample in formatted_samples:
            f.write(json.dumps(sample) + '\n')
    print(f"  Saved: {formatted_output}")

    # Also save noformat version (like noformat - just raw text)
    noformat_output = OUTPUT_DIR / "entertainment_500_noformat.jsonl"
    with open(noformat_output, 'w') as f:
        for i, ent_sample in enumerate(entertainment_data):
            noformat = {
                "instruction": ent_sample["text"][:4000],  # Truncate to reasonable length
                "input": "",
                "output": "",
                "doc_id": ent_sample["id"]
            }
            f.write(json.dumps(noformat) + '\n')
    print(f"  Saved: {noformat_output}")

    # Show sample
    print("\n" + "=" * 60)
    print("Sample formatted entry:")
    print("=" * 60)
    sample = formatted_samples[0]
    print(f"Doc ID: {sample['doc_id']}")
    print(f"Question: {sample['question']}")
    print(f"Source: {sample['original_source']}")
    print(f"\nInstruction (first 500 chars):")
    print(sample['instruction'][:500] + "...")
    print(f"\nOutput: {sample['output'][:200]}...")

    print("\n" + "=" * 60)
    print("Done! Files created:")
    print(f"  - {formatted_output} (with QA template)")
    print(f"  - {noformat_output} (raw text only)")
    print("=" * 60)


if __name__ == "__main__":
    main()
