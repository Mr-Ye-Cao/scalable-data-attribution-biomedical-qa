#!/bin/bash
# Run pretrain gradient computation for ckpt32 and ckpt96 sequentially
#
# Usage: ./scripts/attribution/run_pretrain_ckpt32_ckpt96.sh
#
# Estimated time: ~8 hours each = ~16 hours total

set -e

cd /lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pubmed-gh200

export PYTHONPATH=".:./RapidIn:$PYTHONPATH"

echo "============================================================"
echo "Running Pretrain Gradient Computation: ckpt32 + ckpt96"
echo "Started: $(date)"
echo "============================================================"

# Create output directories
mkdir -p results/rapidin_original/grads/ckpt32/pretrain
mkdir -p results/rapidin_original/grads/ckpt96/pretrain

# ============================================================
# Job 1: Checkpoint-32 Pretrain (~8 hours)
# ============================================================
echo ""
echo "[1/2] Checkpoint-32 - Pretrain (18,353 samples)"
echo "Started: $(date)"
python scripts/attribution/run_rapidin_original.py --config configs/pretrain/ckpt32.json 2>&1 | tee results/rapidin_original/ckpt32_pretrain.log
echo "Completed: $(date)"

# ============================================================
# Job 2: Checkpoint-96 Pretrain (~8 hours)
# ============================================================
echo ""
echo "[2/2] Checkpoint-96 - Pretrain (18,353 samples)"
echo "Started: $(date)"
python scripts/attribution/run_rapidin_original.py --config configs/pretrain/ckpt96.json 2>&1 | tee results/rapidin_original/ckpt96_pretrain.log
echo "Completed: $(date)"

echo ""
echo "============================================================"
echo "ALL PRETRAIN JOBS COMPLETED!"
echo "Finished: $(date)"
echo "============================================================"
echo ""
echo "Gradient directories:"
echo "  - results/rapidin_original/grads/ckpt32/pretrain/"
echo "  - results/rapidin_original/grads/ckpt96/pretrain/"
