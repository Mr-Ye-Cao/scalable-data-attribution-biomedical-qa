#!/usr/bin/env python3
"""
Prepare data files for original RapidIn.

Format: JSONL with {"instruction": "...", "input": "", "output": "..."}
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PUBMEDQA_FILE = PROJECT_ROOT / "pubmedqa" / "data" / "ori_pqal.json"
TEST_GT_FILE = PROJECT_ROOT / "pubmedqa" / "data" / "test_ground_truth.json"
BM25_CANDIDATES_FILE = PROJECT_ROOT / "data" / "bm25_candidates" / "candidates_0013_top1000.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "rapidin_original"


def load_pubmedqa():
    """Load PubMedQA data."""
    with open(PUBMEDQA_FILE) as f:
        return json.load(f)


def load_test_split():
    """Load official test split."""
    with open(TEST_GT_FILE) as f:
        return json.load(f)


def load_bm25_candidates():
    """Load BM25 candidates."""
    with open(BM25_CANDIDATES_FILE) as f:
        return json.load(f)


def prepare_finetune_data():
    """Prepare fine-tuning data (500 train samples) in RapidIn format.

    Format: {context}\n{question} {answer}
    This avoids format bias from QA templates.
    """
    pubmedqa = load_pubmedqa()
    test_split = load_test_split()
    test_pubids = set(test_split.keys())

    output_file = OUTPUT_DIR / "finetune_500.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    for pubid, data in pubmedqa.items():
        if pubid in test_pubids:
            continue  # Skip test samples

        context = " ".join(data.get("CONTEXTS", []))
        question = data.get("QUESTION", "")
        answer = data.get("final_decision", "")

        # Format: raw text without QA template
        text = f"{context}\n{question} {answer}"

        samples.append({
            "instruction": text,
            "input": "",
            "output": "",  # Empty output for pure text loss
            "pubid": pubid
        })

    print(f"Prepared {len(samples)} fine-tuning samples")

    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    print(f"Saved to {output_file}")
    return samples


def prepare_test_data():
    """Prepare test data (500 test samples) in RapidIn format.

    Format: Question: {question}\nAnswer: {answer}
    This is the format used during evaluation.
    """
    pubmedqa = load_pubmedqa()
    test_split = load_test_split()

    output_file = OUTPUT_DIR / "test_500.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    for pubid, answer in test_split.items():
        if pubid not in pubmedqa:
            continue

        data = pubmedqa[pubid]
        question = data.get("QUESTION", "")

        # Format: QA template (matching evaluation format)
        instruction = f"Question: {question}"
        output = answer

        samples.append({
            "instruction": instruction,
            "input": "",
            "output": output,
            "pubid": pubid
        })

    print(f"Prepared {len(samples)} test samples")

    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    print(f"Saved to {output_file}")
    return samples


def prepare_pretrain_candidates(top_k=50):
    """Prepare pre-training candidates in RapidIn format.

    Collects unique candidates from TF-IDF retrieval (top K per query).
    Only uses TEST queries (500) for attribution, not all 1000 PubMedQA.
    Format: raw text

    Args:
        top_k: Number of top candidates per query (default: 50)
    """
    bm25_data = load_bm25_candidates()
    test_split = load_test_split()
    test_pubids = set(test_split.keys())

    output_file = OUTPUT_DIR / f"pretrain_top{top_k}.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Collect unique candidates (only for TEST queries)
    unique_docs = {}
    test_query_count = 0
    for query_data in bm25_data:
        # Only use test queries for attribution
        pubid = str(query_data.get("pubid", ""))
        if pubid not in test_pubids:
            continue

        test_query_count += 1
        candidates = query_data.get("candidates", [])[:top_k]  # Top K per query
        for cand in candidates:
            doc_id = cand["doc_id"]
            if doc_id not in unique_docs:
                # Key is "text_preview" not "text"
                text = cand.get("text_preview", cand.get("text", ""))
                unique_docs[doc_id] = text[:4000]  # Truncate to 4000 chars

    print(f"Used {test_query_count} test queries (filtered from {len(bm25_data)} total)")
    print(f"Collected {len(unique_docs)} unique pre-training candidates (top-{top_k}/query)")

    samples = []
    for doc_id, text in unique_docs.items():
        samples.append({
            "instruction": text,
            "input": "",
            "output": "",  # Empty output for pure text loss
            "doc_id": doc_id
        })

    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    print(f"Saved to {output_file}")

    # Also save the doc_id to index mapping
    mapping = {sample["doc_id"]: i for i, sample in enumerate(samples)}
    mapping_file = OUTPUT_DIR / f"pretrain_top{top_k}_doc_id_to_idx.json"
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f)
    print(f"Saved mapping to {mapping_file}")

    return samples


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare data for original RapidIn")
    parser.add_argument("--top-k", type=int, default=50, help="Top K candidates per query (default: 50)")
    parser.add_argument("--pretrain-only", action="store_true", help="Only prepare pre-training candidates")
    args = parser.parse_args()

    print("=" * 60)
    print("Preparing data for original RapidIn")
    print("=" * 60)

    if not args.pretrain_only:
        print("\n[1/3] Preparing fine-tuning data...")
        prepare_finetune_data()

        print("\n[2/3] Preparing test data...")
        prepare_test_data()

        print("\n[3/3] Preparing pre-training candidates...")
    else:
        print("\nPreparing pre-training candidates only...")

    prepare_pretrain_candidates(top_k=args.top_k)

    print("\n" + "=" * 60)
    print("Done! Data saved to:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
