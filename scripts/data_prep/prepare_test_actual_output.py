#!/usr/bin/env python3
"""
Prepare test data using ACTUAL MODEL OUTPUTS (not ground truth).

This measures: "Which training samples influenced what the model ACTUALLY predicted?"

Creates separate test files for baseline and ckpt64 since they have different outputs.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
EVAL_DIR = PROJECT_ROOT / "eval_output_results"
OUTPUT_DIR = PROJECT_ROOT / "data" / "rapidin_aligned"


def prepare_test_from_eval(eval_file: str, output_name: str):
    """
    Create test data from actual model evaluation outputs.

    Args:
        eval_file: Path to *_predictions_full.json
        output_name: Name for output file (e.g., "test_500_baseline_output")
    """
    eval_path = EVAL_DIR / eval_file
    output_file = OUTPUT_DIR / f"{output_name}.jsonl"

    with open(eval_path) as f:
        eval_data = json.load(f)

    print(f"Loading {len(eval_data)} samples from {eval_file}")

    samples = []
    for pubid, data in eval_data.items():
        # instruction = the prompt given to model (ends with <|im_start|>assistant\n)
        instruction = data['input_prompt']

        # output = what the model actually generated
        # Note: model_output may not have <|im_end|> - we add it for consistency
        model_output = data['model_output']
        if not model_output.strip().endswith('<|im_end|>'):
            model_output = model_output.rstrip() + '<|im_end|>'

        samples.append({
            "instruction": instruction,
            "output": model_output,
            "pubid": pubid,
            "question": data['question'],
            "decision": data['decision'],  # Model's prediction
            "ground_truth": data['ground_truth'],
            "correct": data['correct'],
        })

    # Save
    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"Saved {len(samples)} samples to {output_file}")

    # Stats
    correct_count = sum(1 for s in samples if s['correct'])
    print(f"  Correct predictions: {correct_count}/{len(samples)} ({100*correct_count/len(samples):.1f}%)")

    # Save human-readable sample
    sample_file = OUTPUT_DIR / f"{output_name}_sample.txt"
    with open(sample_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"TEST DATA WITH ACTUAL MODEL OUTPUT: {output_name}\n")
        f.write("=" * 80 + "\n")
        f.write("\nGradients computed on what the model ACTUALLY predicted.\n")
        f.write("This measures which training data influenced the model's behavior.\n\n")

        for i, sample in enumerate(samples[:3]):
            f.write(f"--- Sample {i+1} (pubid: {sample['pubid']}) ---\n")
            f.write(f"Decision: {sample['decision']} (GT: {sample['ground_truth']}, Correct: {sample['correct']})\n\n")
            f.write(f"INSTRUCTION:\n{sample['instruction'][:500]}...\n\n")
            f.write(f"OUTPUT (actual model response):\n{sample['output'][:500]}...\n\n")

    print(f"Saved sample to {sample_file}")
    return samples


def main():
    print("=" * 80)
    print("Preparing test data with ACTUAL MODEL OUTPUTS")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Baseline model outputs...")
    prepare_test_from_eval(
        "baseline_predictions_full.json",
        "test_500_baseline_output"
    )

    print("\n[2/3] Checkpoint-32 (epoch 1) model outputs...")
    prepare_test_from_eval(
        "epoch1_checkpoint-32_predictions_full.json",
        "test_500_ckpt32_output"
    )

    print("\n[3/4] Checkpoint-64 (epoch 2) model outputs...")
    prepare_test_from_eval(
        "epoch2_checkpoint-64_predictions_full.json",
        "test_500_ckpt64_output"
    )

    print("\n[4/4] Checkpoint-96 (epoch 3) model outputs...")
    prepare_test_from_eval(
        "epoch3_final_predictions_full.json",
        "test_500_ckpt96_output"
    )

    print("\n" + "=" * 80)
    print("Done! Files created:")
    print("  - test_500_baseline_output.jsonl")
    print("  - test_500_ckpt32_output.jsonl")
    print("  - test_500_ckpt64_output.jsonl")
    print("  - test_500_ckpt96_output.jsonl")
    print("=" * 80)


if __name__ == "__main__":
    main()
