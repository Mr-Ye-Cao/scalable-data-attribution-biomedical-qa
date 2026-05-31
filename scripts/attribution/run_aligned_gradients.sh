#!/bin/bash
# Run RapidIn gradient caching for aligned data
#
# Usage: ./scripts/attribution/run_aligned_gradients.sh
#
# This will cache gradients for:
# 1. Baseline model + finetune data
# 2. Baseline model + test data
# 3. Checkpoint-64 model + finetune data
# 4. Checkpoint-64 model + test data

set -e

cd /lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pubmed-gh200

echo "========================================"
echo "Running RapidIn with ALIGNED data"
echo "========================================"

# Create output directories
mkdir -p results/rapidin_aligned/grads/baseline/finetune
mkdir -p results/rapidin_aligned/grads/baseline/test
mkdir -p results/rapidin_aligned/grads/ckpt64/finetune
mkdir -p results/rapidin_aligned/grads/ckpt64/test

echo ""
echo "[1/4] Baseline model + Finetune data..."
python scripts/attribution/run_rapidin_original.py --config configs/aligned/baseline_finetune.json 2>&1 | tee results/rapidin_aligned/baseline_finetune.log

echo ""
echo "[2/4] Baseline model + Test data..."
python scripts/attribution/run_rapidin_original.py --config configs/aligned/baseline_test.json 2>&1 | tee results/rapidin_aligned/baseline_test.log

echo ""
echo "[3/4] Checkpoint-64 model + Finetune data..."
python scripts/attribution/run_rapidin_original.py --config configs/aligned/ckpt64_finetune.json 2>&1 | tee results/rapidin_aligned/ckpt64_finetune.log

echo ""
echo "[4/4] Checkpoint-64 model + Test data..."
python scripts/attribution/run_rapidin_original.py --config configs/aligned/ckpt64_test.json 2>&1 | tee results/rapidin_aligned/ckpt64_test.log

echo ""
echo "========================================"
echo "All gradients cached!"
echo "========================================"
echo ""
echo "Next step: Run influence comparison"
echo "  python scripts/attribution/compare_aligned_influence.py"
