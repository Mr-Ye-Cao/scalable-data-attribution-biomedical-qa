# Data Aggregate Documentation

This folder contains all data and computed results for a training data attribution study on a medical question-answering task.

**Created**: 2026-01-15
**Last Updated**: 2026-01-20

---

## Research Objective

**Task**: Measure and compare the influence of different training data sources on a language model's predictions for medical question-answering.

**Research Questions**:
1. How much does fine-tuning data (medical QA) influence model predictions compared to pretraining data?
2. How much of the measured influence comes from the QA format structure vs. the actual content/knowledge?
3. How does the influence of different data sources change as the model is fine-tuned?

**Method**: Gradient-based training data attribution (RapidIn/TracIn) using cosine similarity between gradients.

---

## Model Checkpoints

The study tracks a single model across 6 training stages:

| Checkpoint | Description | Training Stage | Test Accuracy |
|------------|-------------|----------------|---------------|
| `baseline` | OLMo-3-7B-Instruct | Before fine-tuning | 41.8% |
| `ckpt32` | Checkpoint at step 32 | After Epoch 1 | 69.0% |
| `ckpt64` | Checkpoint at step 64 | After Epoch 2 | 74.4% |
| `ckpt96` | Checkpoint at step 96 | After Epoch 3 | 69.0% |
| `ckpt128` | Checkpoint at step 128 | After Epoch 4 | TBD |
| `ckpt160` | Checkpoint at step 160 | After Epoch 5 | TBD |

- **Base Model**: OLMo-3-7B-Instruct (7 billion parameters)
- **Fine-tuning Task**: PubMedQA (medical yes/no/maybe classification)
- **Fine-tuning Data**: 500 training samples from PubMedQA

---

## Data Sources

### 1. Finetune Data (Medical QA)
- **Source**: PubMedQA labeled dataset
- **Content**: Medical research abstracts with yes/no/maybe questions
- **Samples**: 500
- **Format**: QA template with ChatML formatting

### 2. Medical Pretrain Data
- **Source**: BM25-retrieved documents from model's pretraining corpus
- **Content**: Medical/scientific documents semantically similar to test queries
- **Samples**: 500
- **Purpose**: Measure influence of domain-relevant pretraining data

### 3. Entertainment Pretrain Data (Control Group)
- **Source**: Random non-medical documents from model's pretraining corpus
- **Content**: Song lyrics, web content, general entertainment text
- **Samples**: 500
- **Purpose**: Control group to isolate format effects from content effects

### 4. Test Data
- **Source**: PubMedQA official test split
- **Content**: Medical questions with ground truth labels
- **Samples**: 500

---

## Format Variations

Each pretrain data source has multiple format versions to study the effect of QA formatting:

### NoFormat (Raw Text)
- Raw text without any QA structure
- Tests influence from content/knowledge alone

### Formatted (QA Template)
- Same content wrapped in ChatML QA template
- Matches the format of finetune and test data
- Tests combined influence of format + content

**QA Template Structure**:
```
<|im_start|>system
You are a clinical expert. Your task is to analyze the given medical literature context and then provide a Final Decision and a Long Answer.
Context: [document text]
<|im_end|>
<|im_start|>user
[question]
<|im_end|>
<|im_start|>assistant
[answer]
<|im_end|>
```

---

## Folder Structure

```
data-aggregate/
├── input/                              # Raw input data files
│   ├── finetune/
│   │   └── finetune_500.jsonl          # Fine-tuning data (500 medical QA)
│   ├── pretrain/
│   │   ├── pretrain_500_raw.jsonl      # Medical pretrain - original documents
│   │   ├── pretrain_500_noformat.jsonl # Medical pretrain - raw text only
│   │   ├── pretrain_500_iter1.jsonl    # Medical pretrain - QA formatted (all "maybe")
│   │   ├── pretrain_500_iter2.jsonl    # Medical pretrain - QA formatted (label-matched)
│   │   ├── pretrain_500_iter3.jsonl    # Medical pretrain - QA formatted (full output match)
│   │   ├── pretrain_500_iter4.jsonl    # Medical pretrain - QA formatted (no linebreaks)
│   │   ├── entertainment_500_raw.jsonl       # Entertainment - original documents
│   │   ├── entertainment_500_noformat.jsonl  # Entertainment - raw text only
│   │   └── entertainment_500_formatted.jsonl # Entertainment - QA formatted
│   └── test/
│       ├── test_500.jsonl                  # Test queries
│       ├── test_500_baseline_output.jsonl  # Model outputs at baseline
│       ├── test_500_ckpt32_output.jsonl    # Model outputs at epoch 1
│       ├── test_500_ckpt64_output.jsonl    # Model outputs at epoch 2
│       ├── test_500_ckpt96_output.jsonl    # Model outputs at epoch 3
│       ├── test_500_ckpt128_output.jsonl   # Model outputs at epoch 4
│       └── test_500_ckpt160_output.jsonl   # Model outputs at epoch 5
├── outputs/
│   ├── analysis/                       # Summary statistics and figures
│   │   ├── comprehensive_results.json
│   │   ├── entertainment_comparison_all_checkpoints.json
│   │   ├── distribution_stats.json
│   │   ├── correctness_category_analysis.json
│   │   ├── ft_wins_analysis.json
│   │   ├── tracein_efficiency_comparison/           # GOAL3 Exp 1: TracIn vs RapidIn efficiency
│   │   │   └── tracin_memory_summary.json
│   │   ├── bm25_accuracy_comparison/             # GOAL3 Exp 2: RapidIn vs BM25 accuracy
│   │   │   ├── bm25_attribution_top10.jsonl
│   │   │   ├── bm25_stats_top10.json
│   │   │   └── rapidin_vs_bm25_comparison.json
│   │   └── figures/                    # Visualization plots
│   └── influence_scores/               # Per-model influence score results
│       ├── baseline/
│       ├── ckpt32/
│       ├── ckpt64/
│       ├── ckpt96/
│       ├── ckpt128/
│       └── ckpt160/
└── DATA.md                             # This documentation
```

---

## Input Data Formats

### JSONL Format (All Input Files)

Each line is a JSON object with these fields:

**Finetune/Test Data**:
```json
{
  "instruction": "<full prompt with context and question>",
  "output": "<model answer>",
  "pubid": "<PubMed ID>",
  "question": "<the question text>",
  "final_decision": "yes|no|maybe",
  "ground_truth": "yes|no|maybe"
}
```

**Pretrain Data (NoFormat)**:
```json
{
  "instruction": "<raw document text>",
  "input": "",
  "output": "",
  "doc_id": "<unique document identifier>"
}
```

**Pretrain Data (Formatted)**:
```json
{
  "instruction": "<QA-formatted prompt with document as context>",
  "input": "",
  "output": "<synthetic answer>",
  "doc_id": "<unique document identifier>",
  "question": "<question borrowed from test set>",
  "final_decision": "maybe|yes|no",
  "source": "medical|entertainment"
}
```

---

## Output Data Formats

### Influence Scores JSON

Located in `outputs/influence_scores/{checkpoint}/{dataset}.json`

```json
{
  "model": "baseline|ckpt32|ckpt64|ckpt96|ckpt128|ckpt160",
  "dataset": "<dataset name>",
  "n_test": 500,
  "n_train": 500,
  "summary": {
    "mean_max": 0.482,
    "std_max": 0.029,
    "min_max": 0.358,
    "max_max": 0.546,
    "median_max": 0.484
  },
  "per_query": [
    {
      "test_idx": 0,
      "max_influence": 0.446,
      "top_10_indices": [410, 434, 263, ...],
      "top_10_scores": [0.446, 0.443, 0.442, ...]
    },
    ...
  ]
}
```

**Fields**:
- `mean_max`: Average of the maximum influence score per test query
- `max_influence`: Highest influence score from any training sample for this test query
- `top_10_indices`: Indices of the 10 most influential training samples
- `top_10_scores`: Corresponding influence scores (cosine similarity, range -1 to 1)

### Entertainment Comparison JSON

Located in `outputs/analysis/entertainment_comparison_all_checkpoints.json`

```json
{
  "timestamp": "...",
  "checkpoints": {
    "baseline": {
      "entertainment_formatted": 0.384,
      "entertainment_noformat": 0.103,
      "medical_formatted": 0.402,
      "medical_noformat": 0.109,
      "format_effect": 3.71
    },
    "ckpt32": { ... },
    "ckpt64": { ... },
    "ckpt96": { ... },
    "ckpt128": { ... },
    "ckpt160": { ... }
  }
}
```

**Format Effect**: Ratio of formatted to noformat influence (measures QA template contribution)

---

## Influence Score Computation

Influence scores are computed using gradient-based attribution:

1. Compute gradient of loss with respect to model parameters for each training sample
2. Compute gradient for each test sample
3. Calculate cosine similarity between test gradient and each training gradient
4. Higher similarity = higher influence

**Score Interpretation**:
- Scores range from -1 to 1 (cosine similarity)
- Typical values: 0.1 - 0.5
- Higher score indicates the training sample has more influence on the model's prediction for that test query

---

## Available Datasets per Checkpoint

| Dataset | Description | Available For |
|---------|-------------|---------------|
| `finetune` | PubMedQA training data | all checkpoints |
| `noformat` | Medical pretrain (raw text) | all checkpoints |
| `iter1` | Medical pretrain (QA formatted, all "maybe") | baseline, ckpt32, ckpt64, ckpt96, ckpt128, ckpt160 |
| `iter2` | Medical pretrain (QA formatted, label-matched) | baseline, ckpt32, ckpt64 |
| `iter3` | Medical pretrain (QA formatted, full output match) | baseline, ckpt32, ckpt64 |
| `iter4` | Medical pretrain (QA formatted, no linebreaks) | all checkpoints |
| `entertainment` | Entertainment (QA formatted) | all checkpoints |
| `entertainment_noformat` | Entertainment (raw text) | all checkpoints |

---

## Medical Pretrain Format Variations (iter1-iter4)

All versions contain the same 500 BM25-retrieved medical documents, with different formatting:

| Version | Answer Label | Output Format | Context Format |
|---------|--------------|---------------|----------------|
| `iter1` | All "maybe" | Synthetic long answer | Original with linebreaks |
| `iter2` | Matched to source query | Synthetic long answer | Original with linebreaks |
| `iter3` | Matched to source query | Copied from finetune data | Original with linebreaks |
| `iter4` | Matched to source query | Copied from finetune data | Linebreaks removed |

---

## Key Results Summary

### Influence Scores Across All Checkpoints (Mean Max per Test Query)

| Checkpoint | Finetune | Iter1 (Med+Fmt) | Iter4 (Med+Fmt) | NoFormat (Med) | Ent+Fmt | Ent NoFmt | Format Effect |
|------------|----------|-----------------|-----------------|----------------|---------|-----------|---------------|
| **baseline** | 0.4815 | 0.4019 | 0.3946 | 0.1086 | 0.3839 | 0.1035 | 3.71x |
| **ckpt32** | 0.4734 | 0.3968 | 0.4310 | 0.1340 | 0.3200 | 0.0969 | 3.30x |
| **ckpt64** | 0.3862 | 0.3267 | 0.3651 | 0.1173 | 0.2838 | 0.0951 | 2.99x |
| **ckpt96** | 0.3436 | 0.2989 | 0.3268 | 0.1103 | 0.2734 | 0.0911 | 3.00x |
| **ckpt128** | 0.3137 | 0.2694 | 0.2905 | 0.0962 | 0.2345 | 0.0819 | 2.86x |
| **ckpt160** | 0.3661 | 0.3180 | 0.3407 | 0.1093 | 0.2717 | 0.0936 | 2.90x |

### Training Dynamics (Baseline → Epoch 5)

| Data Type | Baseline | Epoch 5 | Change |
|-----------|----------|---------|--------|
| Entertainment + Format | 0.3839 | 0.2717 | ↓29.2% |
| Entertainment NoFormat | 0.1035 | 0.0936 | ↓9.5% |
| Medical + Format | 0.4019 | 0.3180 | ↓20.9% |
| Medical NoFormat | 0.1086 | 0.1093 | ↑0.6% |

### Key Findings

1. **Format effect decreases during training**: 3.71x → 2.86x (entertainment formatted / noformat ratio)
2. **Entertainment influence drops more** (-29.2%) than medical (-20.9%) for formatted data
3. **Medical NoFormat is stable** (+0.6%) while Entertainment NoFormat drops (-9.5%)
4. **Medical > Entertainment** consistently across all epochs (validates attribution method)
5. **Epoch 4 (ckpt128)** shows lowest influence - possible transition point
6. **Epoch 5 (ckpt160)** shows slight recovery in influence scores

---

## BM25 vs RapidIn Comparison (GOAL3 Experiment 2)

**Added**: 2026-01-22

This comparison demonstrates that RapidIn provides more accurate attribution than BM25 because it captures the model's actual learning behavior through gradients.

### Data Used (Same for Both Methods)
- **Finetune**: `finetune_500.jsonl` (500 PubMedQA train samples)
- **Pretrain**: `pretrain_500_iter4.jsonl` (500 BM25-retrieved medical docs)
- **Test**: `test_500.jsonl` (500 test samples)

### Results @ Epoch 2 (Best Model, 74.4% Accuracy)

| Method | FT Wins | PT Wins | Interpretation |
|--------|---------|---------|----------------|
| **RapidIn** | 433 (86.6%) | 67 (13.4%) | Gradient-based: captures model's actual learning |
| **BM25** | 202 (40.4%) | 298 (59.6%) | Lexical-based: ~50-50 based on text similarity |

### Key Finding

The **46.2 percentage point difference** (86.6% - 40.4%) demonstrates that:
- RapidIn correctly identifies that the finetuned model relies more on finetune data
- BM25 cannot distinguish because both FT and PT have similar medical vocabulary
- RapidIn is more accurate for training data attribution

### Output Files
- `bm25_accuracy_comparison/bm25_attribution_top10.jsonl` - Per-query BM25 retrieval results
- `bm25_accuracy_comparison/bm25_stats_top10.json` - BM25 efficiency statistics
- `bm25_accuracy_comparison/rapidin_vs_bm25_comparison.json` - Summary comparison

---

## TracIn Memory Estimation (GOAL3 Experiment 1)

**Added**: 2026-01-22

This analysis demonstrates that TracIn (full gradient storage) is infeasible at LLM scale, justifying RapidIn's compression approach.

### Experimental Setup
- **Model**: OLMo-3-7B-Instruct (7.30B parameters)
- **Method**: Computed full gradients for 3-5 samples, measured actual storage
- **Verified**: TracIn produces valid influence scores (0.19-0.41 range, consistent with RapidIn)

### Storage Requirements

| Method | Per-Sample | 500 Samples | 1000 Samples | Compression |
|--------|------------|-------------|--------------|-------------|
| **TracIn (FP16)** | 13.59 GB | 6.64 TB | 13.28 TB | 1x |
| **RapidIn (K=65536)** | 130 KB | 65 MB | 130 MB | **104,000x** |

### Key Findings

1. **TracIn gradient size**: 7.30B params × 2 bytes (FP16) = **13.59 GB per sample**
2. **Actual file size on disk**: Matches theoretical (13.59 GB)
3. **GPU memory issue**: Cannot store full gradient while model is loaded (model uses 81GB)
4. **Storage infeasibility**: 500 samples require 6.64 TB - exceeds typical storage

### Why RapidIn is Necessary

| Aspect | TracIn | RapidIn |
|--------|--------|---------|
| Storage for 500 samples | 6.64 TB | 65 MB |
| Influence score validity | ✓ Valid (0.19-0.41) | ✓ Valid (0.36-0.55) |
| Practical at LLM scale | ✗ No | ✓ Yes |

### Output Files
- `tracein_efficiency_comparison/tracin_memory_summary.json` - Full measurement results

---

## Key Metrics

- **Mean Max Influence**: Average of maximum influence scores across all test queries
- **FT/PT Ratio**: Finetune influence divided by pretrain influence
- **FT Wins**: Number of test queries where finetune has higher max influence than pretrain
- **Format Effect**: Ratio of formatted to noformat influence (same content, different format)

---

## Notes for Analysis

1. **Influence Score Interpretation**: Higher values indicate stronger attribution to that training data
2. **2x2 Experimental Design**: (Medical vs Entertainment) x (Formatted vs NoFormat) allows isolating format and content effects
3. **Training Dynamics**: Comparing across checkpoints shows how influence changes during fine-tuning
4. **Control Group**: Entertainment data serves as a control - any influence from it comes primarily from format, not domain-relevant content

---

## Original File Locations

For reference, here are the original locations of the source files:

### Pretrain
- `pretrain_500_noformat.jsonl` ← `results/validation/data/pretrain_500_noformat.jsonl`
- `pretrain_500_iter1.jsonl` ← `results/controlled_experiment/data/pretrain_500_aligned.jsonl`
- `pretrain_500_iter4.jsonl` ← `results/controlled_experiment/iteration4/data/pretrain_500_no_linebreaks.jsonl`
- `entertainment_500_*.jsonl` ← `data-aggregate/input/pretrain/entertainment_500_*.jsonl`

### Finetune/Test
- `finetune_500.jsonl` ← `data/rapidin_aligned/finetune_500_aligned.jsonl`
- `test_500.jsonl` ← `data/rapidin_aligned/test_500_aligned.jsonl`
- `test_500_*_output.jsonl` ← `data/rapidin_aligned/test_500_*_output.jsonl`

### Influence Scores
- `outputs/influence_scores/*` ← Computed from gradient files using attribution scripts
  - Test gradients: `results/rapidin_aligned_500/grads/{model}/test/`
  - Finetune gradients: `results/rapidin_aligned_500/grads/{model}/finetune/`
  - Pretrain gradients: `results/validation/grads/{model}/{dataset}/`

---

## Gradient Files (Not Copied)

Gradient files are large (~500MB each) and stored separately:

| Location | Description |
|----------|-------------|
| `results/validation/grads/` | Pretrain gradients (medical and entertainment) |
| `results/rapidin_aligned_500/grads/` | Finetune and test gradients |
