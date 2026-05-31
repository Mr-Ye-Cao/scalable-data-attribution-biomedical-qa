#!/bin/bash
# Run all 19 gradient computations sequentially

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pubmed-gh200

cd /lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune

CONFIG_DIR="configs/new_epochs"

# List of all configs to run (19 total)
CONFIGS=(
    # ckpt96 (5 configs)
    "ckpt96_iter1.json"
    "ckpt96_iter4.json"
    "ckpt96_noformat.json"
    "ckpt96_entertainment.json"
    "ckpt96_entertainment_noformat.json"
    # ckpt128 (7 configs)
    "ckpt128_test.json"
    "ckpt128_finetune.json"
    "ckpt128_iter1.json"
    "ckpt128_iter4.json"
    "ckpt128_noformat.json"
    "ckpt128_entertainment.json"
    "ckpt128_entertainment_noformat.json"
    # ckpt160 (7 configs)
    "ckpt160_test.json"
    "ckpt160_finetune.json"
    "ckpt160_iter1.json"
    "ckpt160_iter4.json"
    "ckpt160_noformat.json"
    "ckpt160_entertainment.json"
    "ckpt160_entertainment_noformat.json"
)

TOTAL=${#CONFIGS[@]}
COUNT=0

echo "=========================================="
echo "Running $TOTAL gradient computations sequentially"
echo "=========================================="
echo ""

for config in "${CONFIGS[@]}"; do
    COUNT=$((COUNT + 1))
    echo ""
    echo "=========================================="
    echo "[$COUNT/$TOTAL] Running: $config"
    echo "Started at: $(date)"
    echo "=========================================="

    python scripts/attribution/run_rapidin_original.py --config "$CONFIG_DIR/$config"

    echo "Completed: $config at $(date)"
done

echo ""
echo "=========================================="
echo "ALL $TOTAL GRADIENT COMPUTATIONS COMPLETED!"
echo "Finished at: $(date)"
echo "=========================================="
