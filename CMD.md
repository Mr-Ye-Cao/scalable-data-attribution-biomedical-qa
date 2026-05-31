# Useful Commands

## Model Inference

Run OLMo-2 1B model inference:

```bash
conda activate pubmed-llm && python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = './OLMo-2-0425-1B'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto')

prompt = 'What is the treatment for hypertension?'
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
"
```

## Download PubMedQA Dataset

Download all PubMedQA subsets (pqa_labeled, pqa_unlabeled, pqa_artificial):

```bash
conda activate pubmed-llm && python scripts/data_prep/download_pubmedqa.py
```

Data saved to: `data/pubmedqa/`

## Evaluate Base Model on PubMedQA

Run inference on PubMedQA test set:

```bash
conda activate pubmed-llm && python scripts/eval/eval_base_model.py
```

Predictions saved to: `results/base_model_predictions.json`

Run official evaluation:

```bash
conda activate pubmed-llm && cd pubmedqa && python evaluation.py ../results/base_model_predictions.json
```

### Baseline Results (OLMo-2-1B)

| Metric | Score |
|--------|-------|
| Accuracy | 0.568 |
| Macro-F1 | 0.313 |

## Download Dolma 3 Health Pre-training Data

Download health-related subsets from OLMo-3's pre-training data (Dolma 3 Mix).

Source: `allenai/dolma3_mix-6T-1025`

### Available Health Subsets

| Subset | Description |
|--------|-------------|
| common_crawl-health-0013 | Health web pages |
| common_crawl-health-0014 | Health web pages |
| common_crawl-health-0015 | Health web pages |
| common_crawl-health-0016 | Health web pages |
| common_crawl-health-0017 | Health web pages |
| common_crawl-health-0018 | Health web pages |
| common_crawl-health-0020 | Health web pages |

### Download Commands

Download all health subsets:

```bash
conda activate pubmed-llm && python scripts/data_prep/download_dolma3_health.py
```

Download specific subsets only:

```bash
conda activate pubmed-llm && python scripts/data_prep/download_dolma3_health.py --subsets 0013 0014
```

List available subsets without downloading:

```bash
conda activate pubmed-llm && python scripts/data_prep/download_dolma3_health.py --list-only
```

Download to custom directory:

```bash
conda activate pubmed-llm && python scripts/data_prep/download_dolma3_health.py --output-dir /path/to/dir
```

Data saved to: `data/dolma3_health/`

### Reading the Data

Files are compressed with zstandard. Install and read:

```bash
pip install zstandard
```

```python
import json
import zstandard as zstd
from pathlib import Path

data_dir = Path('data/dolma3_health/common_crawl-health-0013')
shard_file = list(data_dir.glob('*.jsonl.zst'))[0]

dctx = zstd.ZstdDecompressor()
with open(shard_file, 'rb') as f:
    with dctx.stream_reader(f) as reader:
        for line in reader.read().decode('utf-8').strip().split('\n'):
            record = json.loads(line)
            print(record['id'], record['text'][:100])
```

### Dataset Info

- **Source**: `allenai/dolma3_mix-6T-1025` (OLMo-3 pre-training data)
- **Format**: JSONL compressed with zstandard (.jsonl.zst)
- **Size**: ~500MB per subset (9 shards each)
- **Fields**: `id`, `text`, `metadata`

## Download OLMo-mix-1124 Health Data (pes2o)

Extract health-related content (Medicine, Biology) from the pes2o subset of OLMo-mix-1124.

Source: `allenai/olmo-mix-1124` (OLMo-2 pre-training data)

### Process All 26 pes2o Shards (Parallel - Recommended)

Download and extract health samples from all shards in parallel:

```bash
# Using screen for long-running job
conda activate pubmed-llm
screen -S pes2o_health
python scripts/data_prep/process_all_pes2o_health_parallel.py --workers 4

# Or without screen (foreground)
conda activate pubmed-llm && python scripts/data_prep/process_all_pes2o_health_parallel.py --workers 4
```

**Output:**
- `data/olmo_mix_1124/pes2o-XXXX-health.jsonl` (one file per shard, 0000-0025)
- `data/olmo_mix_1124/pes2o-health-log.txt` (field counts for all shards)
- `data/olmo_mix_1124/pes2o-health-progress.json` (for resume capability)

**Features:**
- Parallel downloads (default 4 workers) to maximize bandwidth
- Resume capability - run again to continue from where it stopped
- Logs field distribution (s2fieldsofstudy) for each shard
- Deletes original .gz files after extraction to save disk space

### Process All Shards (Sequential)

Alternative script that processes one shard at a time:

```bash
conda activate pubmed-llm && python scripts/data_prep/process_all_pes2o_health.py
```

### Extract Single Shard

Process a single downloaded shard:

```bash
# First download a shard manually
wget https://huggingface.co/datasets/allenai/olmo-mix-1124/resolve/main/data/pes2o/pes2o-0025.json.gz -O data/olmo_mix_1124/pes2o-0025.json.gz

# Then extract health samples
conda activate pubmed-llm && python scripts/data_prep/extract_health_from_pes2o.py data/olmo_mix_1124/pes2o-0025.json.gz data/olmo_mix_1124/pes2o-0025-health.jsonl
```

### Monitor Processing Progress

Check status while processing runs:

```bash
# Check output file sizes
ls -lh data/olmo_mix_1124/pes2o-*-health.jsonl

# Check progress file
cat data/olmo_mix_1124/pes2o-health-progress.json

# Check log file
tail -50 data/olmo_mix_1124/pes2o-health-log.txt

# Attach to screen session
screen -r pes2o_health
# Detach with Ctrl+A, D
```

### Dataset Info

- **Source**: `allenai/olmo-mix-1124` (pes2o subset)
- **Total Shards**: 26 (pes2o-0000 to pes2o-0025)
- **Shard Size**: ~4GB compressed, ~15GB uncompressed
- **Health Fields**: Medicine, Biology (filtered by `s2fieldsofstudy` metadata)
- **Health Rate**: ~46% of samples have Medicine or Biology tags
- **Format**: JSONL (JSON lines)
- **Fields**: `id`, `text`, `metadata` (includes `s2fieldsofstudy`)

## OLMo-2-1B Gradient Benchmarking

Benchmark gradient computation speed for OLMo-2-1B (used for RapidIn attribution).

```bash
conda activate pubmed-llm && python scripts/eval/benchmark_olmo2_gradient.py
```

### Benchmark Results (RTX 5090)

| Operation | Time |
|-----------|------|
| Model size | 1.48B params |
| GPU memory | 3 GB |
| Gradient computation | 0.028s/sample |
| RapidGrad compression (K=65536) | 0.37s/sample |
| Compressed dot product | 4.4 μs/pair |

### Comparison with OLMo-3-7B

| Model | Gradient Time | Speedup |
|-------|---------------|---------|
| OLMo-3-7B (CPU offload) | 28.6s | 1x |
| OLMo-2-1B (GPU only) | 0.028s | **1000x** |

## Fine-tune OLMo-2-1B on PubMedQA

Fine-tune OLMo-2-1B on PubMedQA labeled dataset via Medical-LLM-Fine-tuning framework.

### LoRA Fine-tuning (Default)

```bash
conda activate pubmed-llm && cd Medical-LLM-Fine-tuning && python train.py --model_type olmo2-1b --epochs 3
```

### Full Fine-tuning

```bash
conda activate pubmed-llm && cd Medical-LLM-Fine-tuning && python train.py --model_type olmo2-1b --epochs 3 --full_finetune
```

### Run in Screen Session (Recommended)

```bash
# LoRA
screen -S olmo2_lora
conda activate pubmed-llm && cd Medical-LLM-Fine-tuning && python train.py --model_type olmo2-1b --epochs 3
# Detach: Ctrl+A, D

# Full fine-tune
screen -S olmo2_full
conda activate pubmed-llm && cd Medical-LLM-Fine-tuning && python train.py --model_type olmo2-1b --epochs 3 --full_finetune
# Detach: Ctrl+A, D
```

### Configuration Options

```bash
# Custom epochs (LoRA)
cd Medical-LLM-Fine-tuning && python train.py --model_type olmo2-1b --epochs 5

# Custom epochs (Full)
cd Medical-LLM-Fine-tuning && python train.py --model_type olmo2-1b --epochs 5 --full_finetune

# Custom output directory
cd Medical-LLM-Fine-tuning && python train.py --model_type olmo2-1b --output_dir ./results/my_model

# Evaluation only
cd Medical-LLM-Fine-tuning && python train.py --model_type olmo2-1b --eval_only

# Resume from checkpoint
cd Medical-LLM-Fine-tuning && python train.py --model_type olmo2-1b --resume_from_checkpoint ./results/olmo2-1b_pubmedqa_lora/checkpoint-100
```

### Output

```
results/olmo2-1b_pubmedqa_lora/
├── checkpoint-100/      # Intermediate checkpoints
├── checkpoint-171/
├── final/               # LoRA adapter weights
└── merged/              # Merged full model

results/olmo2-1b_pubmedqa_full/
├── checkpoint-100/      # Intermediate checkpoints
├── checkpoint-171/
└── final/               # Final full model (2.97GB)
```

### Training Results Comparison (RTX 5090, pqa_labeled, 3 epochs)

| Metric | LoRA | Full Fine-tune |
|--------|------|----------------|
| Training Time | 98 sec | 137 sec |
| Trainable Params | 2.1M (0.14%) | 1,485M (100%) |
| GPU Memory | ~4 GB | ~15 GB |
| Final Loss | 1.739 | 0.587 |
| Token Accuracy | 60.7% | 86.4% |
| Classification Acc | 100% | 100% |
| ROUGE-L | 0.9989 | 0.9827 |

### Time Estimates for Larger Datasets

| Dataset | Samples | LoRA | Full Fine-tune |
|---------|---------|------|----------------|
| pqa_labeled | 1,000 | 98 sec | 2.3 min |
| pqa_artificial | 211,535 | 47 min | 2.5 hrs |
| pqa_unlabeled | 61,249 | 14 min | 45 min |
| All combined | 273,784 | 1.0 hr | 3.3 hrs |

### Training Details

| Parameter | LoRA | Full Fine-tune |
|-----------|------|----------------|
| Base Model | OLMo-2-0425-1B (1.48B params) | OLMo-2-0425-1B (1.48B params) |
| Method | LoRA (r=16, alpha=32) | All parameters |
| Target Modules | q_proj, v_proj | All |
| Training Data | PubMedQA pqa_labeled (~900 train) | PubMedQA pqa_labeled (~900 train) |
| Learning Rate | 2e-4 | 2e-4 |
| Batch Size | 4 × 4 = 16 effective | 4 × 4 = 16 effective |
| Optimizer | paged_adamw_32bit | paged_adamw_32bit |
| Precision | bfloat16 | bfloat16 |

## Baseline vs Fine-tuning Comparison (80/20 Split)

Evaluate baseline model and compare with fine-tuned models on PubMedQA subsets.

### Dataset Splitting

Each PubMedQA subset is split 80/20 (train/test) with seed=42:

| Subset | Train | Test | Has Decision |
|--------|-------|------|--------------|
| pqa_labeled | 800 | 200 | Yes |
| pqa_artificial | 169,015 | 42,254 | Yes |
| pqa_unlabeled | 48,999 | 12,250 | No |

### Evaluate Baseline (No Fine-tuning)

```bash
# pqa_labeled (smallest, for quick verification)
conda activate pubmed-llm && cd Medical-LLM-Fine-tuning && python eval_baseline.py --model_type olmo2-1b --subset pqa_labeled

# pqa_artificial
conda activate pubmed-llm && cd Medical-LLM-Fine-tuning && python eval_baseline.py --model_type olmo2-1b --subset pqa_artificial

# pqa_unlabeled (no classification metrics, only ROUGE-L)
conda activate pubmed-llm && cd Medical-LLM-Fine-tuning && python eval_baseline.py --model_type olmo2-1b --subset pqa_unlabeled
```

### Fine-tune on Specific Subset

```bash
# LoRA fine-tuning on pqa_labeled
conda activate pubmed-llm && cd Medical-LLM-Fine-tuning && python train.py --model_type olmo2-1b --subset pqa_labeled --epochs 3

# Full fine-tuning on pqa_labeled
conda activate pubmed-llm && cd Medical-LLM-Fine-tuning && python train.py --model_type olmo2-1b --subset pqa_labeled --epochs 3 --full_finetune

# LoRA fine-tuning on pqa_artificial
conda activate pubmed-llm && cd Medical-LLM-Fine-tuning && python train.py --model_type olmo2-1b --subset pqa_artificial --epochs 3
```

### Output Directories

Output directories are auto-named based on model, subset, and method:

```
results/olmo2-1b_pqa_labeled_lora/
├── checkpoint-*/        # Intermediate checkpoints
├── final/               # LoRA adapter weights
└── merged/              # Merged full model

results/olmo2-1b_pqa_labeled_full/
├── checkpoint-*/        # Intermediate checkpoints
└── final/               # Final full model
```

### Results: pqa_labeled (OLMo-2-1B, RTX 5090)

| Method | Accuracy | Macro F1 | ROUGE-L | Train Loss | Time |
|--------|----------|----------|---------|------------|------|
| Baseline (no fine-tuning) | 12.5% | 0.074 | 0.219 | - | - |
| LoRA (r=16, α=32) | 100% | 1.000 | 0.994 | 1.74 | 98s |
| Full Fine-tuning | 100% | 1.000 | 0.983 | 0.59 | 137s |

**Note:** Baseline predicts "maybe" for all samples. 100% accuracy after fine-tuning suggests possible overfitting on the small dataset (800 train samples).

## Fine-tune OLMo-3-7B-Instruct on PubMedQA (GH200)

Full fine-tuning of OLMo-3-7B-Instruct on PubMedQA using official 500/500 train/test split.

### Full Fine-tuning with Official Split (Recommended)

```bash
# Activate GH200 environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pubmed-gh200

# Full fine-tuning with checkpoints saved at each epoch
cd Medical-LLM-Fine-tuning
python train.py \
    --model_type olmo3-7b-instruct \
    --subset pqa_labeled \
    --official_split \
    --full_finetune \
    --epochs 3 \
    --save_every_epoch \
    --save_all_checkpoints \
    --output_dir ./results/olmo3-7b-instruct_official_full
```

### Evaluate on Official Test Set

```bash
# Evaluate baseline
python eval_official.py --model_type olmo3-7b-instruct

# Evaluate fine-tuned checkpoint
python eval_official.py --model_type olmo3-7b-instruct \
    --model_path ./results/olmo3-7b-instruct_official_full/checkpoint-64 \
    --output_file ./results/checkpoint-64_predictions.json

# Verify with official PubMedQA evaluation script
cd ../pubmedqa && python evaluation.py ../Medical-LLM-Fine-tuning/results/checkpoint-64_predictions.json
```

### Results: OLMo-3-7B-Instruct (GH200, Official 500/500 Split)

| Model | Accuracy | Macro-F1 | Notes |
|-------|----------|----------|-------|
| Baseline | 41.8% | 0.407 | Over-predicts "maybe" |
| Epoch 1 (checkpoint-32) | 69.0% | 0.453 | +27.2% |
| **Epoch 2 (checkpoint-64)** | **74.4%** | **0.568** | **Best** (+32.6%) |
| Epoch 3 (checkpoint-96) | 69.0% | 0.453 | Overfitting |

**Best checkpoint:** `results/olmo3-7b-instruct_official_full/checkpoint-64`

### Training Details (GH200)

| Parameter | Value |
|-----------|-------|
| Base Model | OLMo-3-7B-Instruct (7.3B params) |
| Method | Full fine-tuning (all parameters) |
| Training Data | PubMedQA pqa_labeled (500 train, official split) |
| Test Data | PubMedQA official test (500 samples) |
| Learning Rate | 2e-4 |
| Batch Size | 4 × 4 = 16 effective |
| Precision | bfloat16 |
| GPU Memory | 92 GB / 100 GB |
| Training Time | ~6 min (3 epochs) |

### Output Files

```
results/olmo3-7b-instruct_official_full/
├── checkpoint-32/           # Epoch 1
├── checkpoint-64/           # Epoch 2 (Best)
├── checkpoint-96/           # Epoch 3
└── final/                   # Final model (same as epoch 3)

results/
├── baseline_predictions.json                    # Official format (for eval script)
├── baseline_predictions_full.json               # Full format (with reasoning)
├── epoch1_checkpoint-32_predictions.json
├── epoch1_checkpoint-32_predictions_full.json
├── epoch2_checkpoint-64_predictions.json        # Best model
├── epoch2_checkpoint-64_predictions_full.json
├── epoch3_final_predictions.json
└── epoch3_final_predictions_full.json
```

### Output Format

**Official format** (`*_predictions.json`): For official PubMedQA eval script
```json
{"21645374": "yes", "16418930": "no", ...}
```

**Full format** (`*_predictions_full.json`): Complete info with reasoning
```json
{
  "21645374": {
    "pmid": "21645374",
    "question": "Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?",
    "context": "Programmed cell death (PCD) is the regulated death of cells...",
    "input_prompt": "<|im_start|>system\nYou are a clinical expert...",
    "model_output": "Final Decision: yes\nLong Answer: The results indicate...",
    "decision": "yes",
    "reasoning": "The results indicate that the observed changes...",
    "ground_truth": "yes",
    "correct": true
  }
}
```

## Controlled Attribution Experiment

Scripts for the controlled experiment comparing template-matched pretrain vs finetune attribution.

See `GOAL2.md` for full experiment plan.

### Step 1: Strip QA Template from Fine-tuning Data

Remove ChatML tags, system prompt, and prefixes from fine-tuning data to get raw content.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pubmed-gh200
python scripts/attribution/controlled_experiment/strip_qa_template.py
```

**Input:** `data/rapidin_aligned/finetune_500_aligned.jsonl`
**Output:** `results/controlled_experiment/data/finetune_500_raw.jsonl`

**Output format:**
```json
{
  "pubid": "10808977",
  "context": "Telephone counseling and tailored print...",
  "question": "Can tailored interventions increase mammography use among HMO women?",
  "final_decision": "yes",
  "long_answer": "The effects of the intervention were most pronounced..."
}
```

### Step 2: BM25 Retrieval for Pre-training Documents

Run BM25 semantic search to find relevant pre-training documents for each fine-tuning sample.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pubmed-gh200

# Run BM25 search on single subset (0013 = ~264K docs)
python scripts/attribution/controlled_experiment/bm25_retrieve.py --subset 0013 --top-k 10

# Run on all subsets (~1.8M docs) - takes longer
python scripts/attribution/controlled_experiment/bm25_retrieve.py --all-subsets --top-k 10

# Test with small corpus
python scripts/attribution/controlled_experiment/bm25_retrieve.py --subset 0013 --top-k 10 --max-docs 10000
```

**Input:** `results/controlled_experiment/data/finetune_500_raw.jsonl`
**Output:** `results/controlled_experiment/data/bm25_results_{subset}_top{k}.jsonl`

**Output format:**
```json
{
  "pubid": "10808977",
  "question": "Can tailored interventions increase mammography use among HMO women?",
  "context": "Telephone counseling and tailored print... (preview)",
  "final_decision": "yes",
  "candidates": [
    {
      "rank": 1,
      "score": 320.52,
      "doc_id": "doc_123456",
      "shard": "common_crawl-health-0013/shard-001.jsonl.zst",
      "text": "Full document text..."
    },
    ...
  ]
}
```

**Performance:**
- Corpus size: 264K docs (subset 0013)
- Index build time: ~50s
- Search time: ~25s per query
- Total time for 500 queries: ~3.5 hours
