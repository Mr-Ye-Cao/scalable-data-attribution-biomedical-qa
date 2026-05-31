#!/bin/bash
# Run remaining gradient computations for lr2e-5 checkpoints sequentially

CONFIG_DIR="configs/lr2e-5"
SCRIPT="scripts/attribution/run_rapidin_original.py"

# Skip test for ckpt128 (already done)
CONFIGS=(
    "ckpt128_lr2e-5_finetune.json"
    "ckpt128_lr2e-5_iter1.json"
    "ckpt128_lr2e-5_iter4.json"
    "ckpt128_lr2e-5_noformat.json"
    "ckpt128_lr2e-5_entertainment.json"
    "ckpt128_lr2e-5_entertainment_noformat.json"
    "ckpt160_lr2e-5_test.json"
    "ckpt160_lr2e-5_finetune.json"
    "ckpt160_lr2e-5_iter1.json"
    "ckpt160_lr2e-5_iter4.json"
    "ckpt160_lr2e-5_noformat.json"
    "ckpt160_lr2e-5_entertainment.json"
    "ckpt160_lr2e-5_entertainment_noformat.json"
)

echo "Starting remaining gradient computations for lr2e-5 checkpoints..."
echo "Total configs: ${#CONFIGS[@]}"
echo ""

for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    echo "========================================"
    echo "[$(($i + 1))/${#CONFIGS[@]}] Running: $config"
    echo "Time: $(date)"
    echo "========================================"

    python "$SCRIPT" --config "$CONFIG_DIR/$config"

    if [ $? -eq 0 ]; then
        echo "Completed: $config"
    else
        echo "ERROR: Failed on $config"
        exit 1
    fi
    echo ""
done

echo "========================================"
echo "All gradient computations completed!"
echo "Time: $(date)"
echo "========================================"
