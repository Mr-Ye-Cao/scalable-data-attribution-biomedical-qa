#!/bin/bash
#
# Comprehensive RapidIn gradient computation
# 3 models x 5 datasets = 15 runs (14 new + 1 already done)
#
# Uses the CORRECT script: run_rapidin_original.py
#

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pubmed-gh200
cd /lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune

LOG_DIR="./results/validation/logs"
mkdir -p $LOG_DIR

echo "=========================================="
echo "COMPREHENSIVE RAPIDIN GRADIENT COMPUTATION"
echo "Started: $(date)"
echo "=========================================="
echo ""

# Define all configs to run
CONFIGS=(
    "baseline_noformat"
    "baseline_iter2"
    "baseline_iter3"
    "baseline_iter4"
    "ckpt32_noformat"
    "ckpt32_iter1"
    "ckpt32_iter2"
    "ckpt32_iter3"
    "ckpt32_iter4"
    "ckpt64_noformat"
    "ckpt64_iter1"
    "ckpt64_iter2"
    "ckpt64_iter3"
    "ckpt64_iter4"
)

TOTAL=${#CONFIGS[@]}
COUNT=0

for CONFIG_NAME in "${CONFIGS[@]}"; do
    COUNT=$((COUNT + 1))
    CONFIG_PATH="configs/validation/comprehensive/${CONFIG_NAME}.json"
    LOG_FILE="${LOG_DIR}/${CONFIG_NAME}.log"

    echo "=========================================="
    echo "[$COUNT/$TOTAL] Running: $CONFIG_NAME"
    echo "Config: $CONFIG_PATH"
    echo "Started: $(date)"
    echo "=========================================="

    python scripts/attribution/run_rapidin_original.py --config $CONFIG_PATH 2>&1 | tee $LOG_FILE

    echo ""
    echo "Completed: $CONFIG_NAME at $(date)"
    echo ""
done

echo ""
echo "=========================================="
echo "ALL GRADIENT COMPUTATIONS COMPLETE"
echo "Finished: $(date)"
echo "=========================================="
echo ""

# Now compute influence scores
echo "=========================================="
echo "COMPUTING INFLUENCE SCORES"
echo "=========================================="

python scripts/attribution/validation/compute_all_influence.py

echo ""
echo "=========================================="
echo "COMPREHENSIVE RUN COMPLETE"
echo "Finished: $(date)"
echo "=========================================="
