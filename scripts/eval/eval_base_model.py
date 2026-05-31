#!/usr/bin/env python3
"""
Evaluate base OLMo-2 model on PubMedQA test set.

Usage:
    conda activate pubmed-llm
    python scripts/eval_base_model.py
"""

import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "OLMo-2-0425-1B"
DATA_PATH = PROJECT_ROOT / "pubmedqa" / "data" / "ori_pqal.json"
TEST_GT_PATH = PROJECT_ROOT / "pubmedqa" / "data" / "test_ground_truth.json"
RESULTS_DIR = PROJECT_ROOT / "results"


def create_prompt(question: str, contexts: list[str]) -> str:
    """Create prompt for PubMedQA task."""
    context_text = " ".join(contexts)
    prompt = f"""Given the following medical research context and question, first provide your reasoning, then answer with "yes", "no", or "maybe".

Context: {context_text}

Question: {question}

Reasoning and Answer:"""
    return prompt


def extract_answer(response: str) -> str:
    """Extract yes/no/maybe from generated response."""
    response_lower = response.lower()

    # Check for explicit answer patterns
    for pattern in ["answer: yes", "answer is yes", "the answer is yes", "yes."]:
        if pattern in response_lower:
            return "yes"
    for pattern in ["answer: no", "answer is no", "the answer is no", "no."]:
        if pattern in response_lower:
            return "no"
    for pattern in ["answer: maybe", "answer is maybe", "the answer is maybe", "maybe."]:
        if pattern in response_lower:
            return "maybe"

    # Check last sentence or word
    words = response_lower.split()
    if words:
        last_words = " ".join(words[-5:])
        if "yes" in last_words:
            return "yes"
        elif "no" in last_words:
            return "no"
        elif "maybe" in last_words:
            return "maybe"

    # Check first word
    first_word = words[0].strip(".,!?\"'") if words else ""
    if first_word in ["yes", "no", "maybe"]:
        return first_word

    # Fallback: check anywhere
    if "yes" in response_lower[:100]:
        return "yes"
    elif "no" in response_lower[:100]:
        return "no"
    elif "maybe" in response_lower[:100]:
        return "maybe"

    return "maybe"  # Default fallback


def main():
    # Timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("Evaluating Base OLMo-2 on PubMedQA")
    print(f"Timestamp: {timestamp}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    with open(DATA_PATH) as f:
        pqa_data = json.load(f)
    with open(TEST_GT_PATH) as f:
        test_gt = json.load(f)

    print(f"  Total PQA-L examples: {len(pqa_data)}")
    print(f"  Test set size: {len(test_gt)}")

    # Load model
    print("\nLoading OLMo-2 model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    print(f"  Model loaded on: {next(model.parameters()).device}")

    # Prepare output paths
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions_path = RESULTS_DIR / f"base_model_predictions_{timestamp}.json"
    logs_path = RESULTS_DIR / f"base_model_logs_{timestamp}.jsonl"

    # Run inference
    print("\nRunning inference on test set...")
    predictions = {}
    logs = []

    with open(logs_path, "w") as log_file:
        for pmid in tqdm(test_gt.keys(), desc="Evaluating"):
            # Get question and context
            example = pqa_data[pmid]
            question = example["QUESTION"]
            contexts = example["CONTEXTS"]
            ground_truth = test_gt[pmid]

            # Create prompt
            prompt = create_prompt(question, contexts)

            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Generate with more tokens for reasoning
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()

            # Extract answer
            answer = extract_answer(response)
            predictions[pmid] = answer

            # Create log entry
            log_entry = {
                "pmid": pmid,
                "question": question,
                "contexts": contexts,
                "prompt": prompt,
                "model_response": response,
                "extracted_answer": answer,
                "ground_truth": ground_truth,
                "correct": answer == ground_truth
            }
            logs.append(log_entry)

            # Write log entry immediately (streaming) - pretty printed for readability
            log_file.write(json.dumps(log_entry, indent=2) + "\n\n")

    # Save predictions
    with open(predictions_path, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"\nPredictions saved to: {predictions_path}")
    print(f"Logs saved to: {logs_path}")

    # Quick accuracy check
    correct = sum(1 for pmid in test_gt if predictions[pmid] == test_gt[pmid])
    accuracy = correct / len(test_gt)
    print(f"\nAccuracy: {correct}/{len(test_gt)} = {accuracy:.4f}")

    # Show prediction distribution
    from collections import Counter
    pred_dist = Counter(predictions.values())
    print(f"Prediction distribution: {dict(pred_dist)}")
    print(f"Ground truth distribution: {dict(Counter(test_gt.values()))}")

    # Show sample logs
    print("\n" + "=" * 60)
    print("Sample Predictions (first 3):")
    print("=" * 60)
    for log in logs[:3]:
        print(f"\n[PMID: {log['pmid']}]")
        print(f"Question: {log['question']}")
        print(f"Model Response: {log['model_response'][:300]}...")
        print(f"Extracted: {log['extracted_answer']} | Ground Truth: {log['ground_truth']} | {'✓' if log['correct'] else '✗'}")

    print("\n" + "=" * 60)
    print("To run official evaluation:")
    print(f"  cd pubmedqa && python evaluation.py ../{predictions_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
