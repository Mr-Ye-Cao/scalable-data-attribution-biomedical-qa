#!/usr/bin/env python3
"""
Prepare ALIGNED data files for fine-tuning and RapidIn attribution - V2.

CRITICAL FIX: RapidIn data loader expects:
  - "instruction": The INPUT prompt (will be MASKED in loss computation)
  - "output": The TARGET response (gradients computed on these tokens)

The data loader does: full_text = instruction + output + eos_token
Then masks labels for tokens in "instruction", keeping only "output" for gradient.

So we must split:
  - instruction = system prompt + context + user question + assistant start
  - output = the model's response (Final Decision + Long Answer)

This ensures gradients are computed on the ANSWER tokens, not the prompt.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PUBMEDQA_FILE = PROJECT_ROOT / "pubmedqa" / "data" / "ori_pqal.json"
TEST_GT_FILE = PROJECT_ROOT / "pubmedqa" / "data" / "test_ground_truth.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "rapidin_aligned"


def load_pubmedqa():
    """Load PubMedQA data."""
    with open(PUBMEDQA_FILE) as f:
        return json.load(f)


def load_test_split():
    """Load official test split."""
    with open(TEST_GT_FILE) as f:
        return json.load(f)


def format_sample(pubid: str, data: dict, answer: str = None) -> tuple[str, str]:
    """
    Format a sample into (instruction, output) for RapidIn.

    Args:
        pubid: PubMedQA ID
        data: PubMedQA data dict
        answer: Override answer (for test data using ground truth)

    Returns:
        (instruction, output) tuple where:
        - instruction: Everything up to and including "<|im_start|>assistant\n"
        - output: The model's response "Final Decision: ...\nLong Answer: ...<|im_end|>"
    """
    context = " ".join(data.get("CONTEXTS", []))
    question = data.get("QUESTION", "")
    final_decision = answer if answer else data.get("final_decision", "")
    long_answer = data.get("LONG_ANSWER", "")

    # INSTRUCTION: Everything the model receives as input (will be masked)
    instruction = f"""<|im_start|>system
You are a clinical expert. Your task is to analyze the given medical literature context and then provide a Final Decision and a Long Answer.
Context: {context}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""

    # OUTPUT: The model's response (gradients computed here)
    output = f"""Final Decision: {final_decision}
Long Answer: {long_answer}<|im_end|>"""

    return instruction, output


def prepare_finetune_data():
    """
    Prepare fine-tuning data (500 train samples) for RapidIn.

    For training data attribution, we want to measure how each training sample
    influences the model. We compute gradients on the training sample's ANSWER.
    """
    pubmedqa = load_pubmedqa()
    test_split = load_test_split()
    test_pubids = set(test_split.keys())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "finetune_500_aligned.jsonl"

    samples = []
    for pubid, data in pubmedqa.items():
        if pubid in test_pubids:
            continue  # Skip test samples

        instruction, output = format_sample(pubid, data)

        samples.append({
            "instruction": instruction,
            "output": output,
            "pubid": pubid,
            "question": data.get("QUESTION", ""),
            "final_decision": data.get("final_decision", ""),
        })

    print(f"Prepared {len(samples)} fine-tuning samples")

    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"Saved to {output_file}")

    # Save human-readable sample
    sample_file = OUTPUT_DIR / "finetune_sample.txt"
    with open(sample_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FINE-TUNING DATA FORMAT (first 3 samples)\n")
        f.write("=" * 80 + "\n")
        f.write("\nRapidIn computes gradients on the OUTPUT tokens (not instruction).\n")
        f.write("This matches how training loss is computed.\n\n")
        for i, sample in enumerate(samples[:3]):
            f.write(f"--- Sample {i+1} (pubid: {sample['pubid']}) ---\n")
            f.write(f"INSTRUCTION (masked in loss):\n{sample['instruction']}\n")
            f.write(f"OUTPUT (gradients computed here):\n{sample['output']}\n\n")
    print(f"Saved sample to {sample_file}")

    return samples


def prepare_test_data():
    """
    Prepare test data (500 test samples) for RapidIn.

    For test data attribution, we want to measure which training samples
    influenced the model's ability to produce the CORRECT answer.
    We use ground truth labels to compute gradients.
    """
    pubmedqa = load_pubmedqa()
    test_split = load_test_split()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "test_500_aligned.jsonl"

    samples = []
    for pubid, answer in test_split.items():
        if pubid not in pubmedqa:
            print(f"Warning: pubid {pubid} not found in pubmedqa")
            continue

        data = pubmedqa[pubid]
        instruction, output = format_sample(pubid, data, answer=answer)

        samples.append({
            "instruction": instruction,
            "output": output,
            "pubid": pubid,
            "question": data.get("QUESTION", ""),
            "ground_truth": answer,
        })

    print(f"Prepared {len(samples)} test samples")

    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"Saved to {output_file}")

    # Save human-readable sample
    sample_file = OUTPUT_DIR / "test_sample.txt"
    with open(sample_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TEST DATA FORMAT (first 3 samples)\n")
        f.write("=" * 80 + "\n")
        f.write("\nRapidIn computes gradients on producing the CORRECT answer.\n")
        f.write("High influence = training sample helped model answer correctly.\n\n")
        for i, sample in enumerate(samples[:3]):
            f.write(f"--- Sample {i+1} (pubid: {sample['pubid']}, GT: {sample['ground_truth']}) ---\n")
            f.write(f"INSTRUCTION (masked in loss):\n{sample['instruction']}\n")
            f.write(f"OUTPUT (gradients computed here):\n{sample['output']}\n\n")
    print(f"Saved sample to {sample_file}")

    return samples


def verify_with_eval_output():
    """Verify alignment with actual evaluation prompts."""
    eval_file = PROJECT_ROOT / "eval_output_results" / "baseline_predictions_full.json"

    if not eval_file.exists():
        print("Warning: Cannot verify - eval output file not found")
        return

    with open(eval_file) as f:
        eval_data = json.load(f)

    # Load our test data
    test_file = OUTPUT_DIR / "test_500_aligned.jsonl"
    our_test = {}
    with open(test_file) as f:
        for line in f:
            sample = json.loads(line)
            our_test[sample['pubid']] = sample

    print("\n" + "=" * 80)
    print("VERIFICATION: Comparing instruction with actual evaluation prompt")
    print("=" * 80)

    sample_pubid = list(eval_data.keys())[0]
    eval_prompt = eval_data[sample_pubid]['input_prompt']
    our_instruction = our_test[sample_pubid]['instruction']

    # The eval prompt should match our instruction (both are the input to model)
    if eval_prompt.strip() == our_instruction.strip():
        print(f"\n✓ INSTRUCTION matches eval prompt for pubid {sample_pubid}")
    else:
        print(f"\n✗ MISMATCH for pubid {sample_pubid}")
        print(f"\nEval prompt length: {len(eval_prompt)}")
        print(f"Our instruction length: {len(our_instruction)}")

        # Find first difference
        for i, (c1, c2) in enumerate(zip(eval_prompt, our_instruction)):
            if c1 != c2:
                print(f"First diff at position {i}: eval={repr(c1)} vs ours={repr(c2)}")
                print(f"Context: ...{eval_prompt[max(0,i-20):i+20]}...")
                break


def create_comparison_file():
    """Create comparison between old and new formats."""
    pubmedqa = load_pubmedqa()
    test_split = load_test_split()

    # Load old data
    old_ft_file = PROJECT_ROOT / "data" / "rapidin_original" / "finetune_500.jsonl"
    old_test_file = PROJECT_ROOT / "data" / "rapidin_original" / "test_500.jsonl"

    comparison_file = OUTPUT_DIR / "format_comparison.txt"
    with open(comparison_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("FORMAT COMPARISON: OLD vs NEW (V2 - Properly Split)\n")
        f.write("=" * 100 + "\n\n")

        f.write("KEY DIFFERENCE:\n")
        f.write("- OLD: Full text in 'instruction', empty 'output' → gradients on wrong tokens!\n")
        f.write("- NEW: Prompt in 'instruction', answer in 'output' → gradients on answer tokens\n\n")

        # Show test example
        test_pubid = list(test_split.keys())[0]

        if old_test_file.exists():
            with open(old_test_file) as of:
                for line in of:
                    old_sample = json.loads(line)
                    if old_sample['pubid'] == test_pubid:
                        break

            f.write("### TEST DATA COMPARISON ###\n\n")
            f.write(f"Sample pubid: {test_pubid}\n\n")

            f.write("OLD FORMAT:\n")
            f.write("-" * 50 + "\n")
            f.write(f"instruction: {old_sample['instruction'][:200]}...\n")
            f.write(f"output: '{old_sample['output']}'\n")
            f.write("PROBLEM: No context! Gradient on 'yes/no' only!\n\n")

            new_instruction, new_output = format_sample(test_pubid, pubmedqa[test_pubid], test_split[test_pubid])
            f.write("NEW FORMAT (V2):\n")
            f.write("-" * 50 + "\n")
            f.write(f"instruction: {new_instruction[:300]}...\n")
            f.write(f"output: {new_output}\n")
            f.write("CORRECT: Full context in instruction, answer in output!\n")

    print(f"Saved comparison to {comparison_file}")


def main():
    print("=" * 80)
    print("Preparing ALIGNED data V2 (properly split instruction/output)")
    print("=" * 80)

    print("\n[1/4] Preparing fine-tuning data (500 train samples)...")
    prepare_finetune_data()

    print("\n[2/4] Preparing test data (500 test samples)...")
    prepare_test_data()

    print("\n[3/4] Verifying alignment with evaluation format...")
    verify_with_eval_output()

    print("\n[4/4] Creating comparison file...")
    create_comparison_file()

    print("\n" + "=" * 80)
    print("Done! Files saved to:", OUTPUT_DIR)
    print("\nKey files:")
    print("  - finetune_500_aligned.jsonl  (500 training samples)")
    print("  - test_500_aligned.jsonl      (500 test samples)")
    print("  - finetune_sample.txt         (human-readable)")
    print("  - test_sample.txt             (human-readable)")
    print("  - format_comparison.txt       (old vs new)")
    print("=" * 80)


if __name__ == "__main__":
    main()
