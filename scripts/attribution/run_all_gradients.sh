#!/bin/bash
# Run all gradient caching jobs sequentially (1 GPU constraint)
# 3 models (baseline, ckpt32, ckpt64) × 3 data types (pretrain, finetune, test)
#
# Usage: ./scripts/attribution/run_all_gradients.sh

set -e  # Exit on error

cd /lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pubmed-gh200

export PYTHONPATH=".:./RapidIn:$PYTHONPATH"

echo "============================================================"
echo "Starting all gradient caching jobs"
echo "Time: $(date)"
echo "============================================================"

# Create output directories
mkdir -p results/rapidin_original/grads/{baseline,ckpt32,ckpt64}/pretrain
mkdir -p results/rapidin_aligned/grads/{baseline,ckpt32,ckpt64}/{finetune,test}

# ============================================================
# PRETRAIN GRADIENTS (~8 hrs each, uses original format)
# ============================================================

echo ""
echo "[1/9] Baseline - Pretrain (18,353 samples)"
echo "Started: $(date)"
python scripts/attribution/run_rapidin_original.py --config configs/pretrain/baseline.json
echo "Completed: $(date)"

echo ""
echo "[2/9] Checkpoint-32 - Pretrain (18,353 samples)"
echo "Started: $(date)"
python scripts/attribution/run_rapidin_original.py --config configs/pretrain/ckpt32.json
echo "Completed: $(date)"

echo ""
echo "[3/9] Checkpoint-64 - Pretrain (18,353 samples)"
echo "Started: $(date)"
python scripts/attribution/run_rapidin_original.py --config configs/pretrain/ckpt64.json
echo "Completed: $(date)"

# ============================================================
# FINETUNE GRADIENTS (~2 min each, uses aligned format)
# ============================================================

echo ""
echo "[4/9] Baseline - Finetune (50 samples)"
echo "Started: $(date)"
python scripts/attribution/run_rapidin_original.py --config configs/aligned/baseline_finetune.json
echo "Completed: $(date)"

echo ""
echo "[5/9] Checkpoint-32 - Finetune (50 samples)"
echo "Started: $(date)"
python scripts/attribution/run_rapidin_original.py --config configs/aligned/ckpt32_finetune.json
echo "Completed: $(date)"

echo ""
echo "[6/9] Checkpoint-64 - Finetune (50 samples)"
echo "Started: $(date)"
python scripts/attribution/run_rapidin_original.py --config configs/aligned/ckpt64_finetune.json
echo "Completed: $(date)"

# ============================================================
# TEST GRADIENTS (~2 min each, uses aligned format)
# ============================================================

echo ""
echo "[7/9] Baseline - Test (50 samples)"
echo "Started: $(date)"
python scripts/attribution/run_rapidin_original.py --config configs/aligned/baseline_test.json
echo "Completed: $(date)"

echo ""
echo "[8/9] Checkpoint-32 - Test (50 samples)"
echo "Started: $(date)"
python scripts/attribution/run_rapidin_original.py --config configs/aligned/ckpt32_test.json
echo "Completed: $(date)"

echo ""
echo "[9/9] Checkpoint-64 - Test (50 samples)"
echo "Started: $(date)"
python scripts/attribution/run_rapidin_original.py --config configs/aligned/ckpt64_test.json
echo "Completed: $(date)"

echo ""
echo "============================================================"
echo "ALL JOBS COMPLETED!"
echo "Time: $(date)"
echo "============================================================"
echo ""
echo "Gradient directories:"
echo "  Pretrain (original format):"
echo "    - results/rapidin_original/grads/{baseline,ckpt32,ckpt64}/pretrain/"
echo "  Finetune/Test (aligned format):"
echo "    - results/rapidin_aligned/grads/{baseline,ckpt32,ckpt64}/{finetune,test}/"
