#!/usr/bin/env python3
"""
Prepare PubMedQA samples for RapidIn test data.
Converts PubMedQA format to instruction/input/output JSONL format.
"""

import json
import sys

def main():
    # Load PubMedQA labeled data
    with open("pubmedqa/data/ori_pqal.json", "r") as f:
        data = json.load(f)

    # Convert to RapidIn format
    samples = []
    for pmid, item in list(data.items())[:5]:  # Take first 5 samples
        context = " ".join(item["CONTEXTS"])
        question = item["QUESTION"]
        answer = item["final_decision"]  # yes/no/maybe

        samples.append({
            "instruction": f"Answer the following medical question based on the context. Answer with yes, no, or maybe.\n\nContext: {context}\n\nQuestion: {question}",
            "input": "",
            "output": answer,
            "metadata": {"pmid": pmid, "source": "pubmedqa"}
        })

    # Save to JSONL
    output_path = "data/rapidin/pubmedqa_test_5.jsonl"
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Saved {len(samples)} samples to {output_path}")

    # Show first sample
    print("\nFirst sample:")
    print(json.dumps(samples[0], indent=2)[:500] + "...")

if __name__ == "__main__":
    main()
