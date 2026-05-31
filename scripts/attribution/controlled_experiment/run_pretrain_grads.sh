#!/bin/bash
# Run pretrain gradient computation for controlled experiment
# Sequential execution to avoid GPU memory conflicts

set -e

echo "=============================================="
echo "Controlled Experiment: Pretrain Gradient Computation"
echo "Started: $(date)"
echo "=============================================="

cd /lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pubmed-gh200

# Status file
STATUS_FILE="/tmp/ctrl_pretrain_status.txt"
echo "Status: Starting" > $STATUS_FILE

# Run baseline
echo ""
echo "=== [1/3] Baseline Pretrain Gradients ==="
echo "Started: $(date)"
echo "Status: Running baseline (1/3)" >> $STATUS_FILE
python RapidIn/MP_main.py --config_path configs/controlled_500/baseline_pretrain.json
echo "Completed: $(date)"
echo "Baseline gradients:" $(ls results/controlled_experiment/grads/baseline/pretrain/*.pt 2>/dev/null | wc -l) "/500"

# Run ckpt32
echo ""
echo "=== [2/3] Checkpoint-32 Pretrain Gradients ==="
echo "Started: $(date)"
echo "Status: Running ckpt32 (2/3)" >> $STATUS_FILE
python RapidIn/MP_main.py --config_path configs/controlled_500/ckpt32_pretrain.json
echo "Completed: $(date)"
echo "Ckpt32 gradients:" $(ls results/controlled_experiment/grads/ckpt32/pretrain/*.pt 2>/dev/null | wc -l) "/500"

# Run ckpt64
echo ""
echo "=== [3/3] Checkpoint-64 Pretrain Gradients ==="
echo "Started: $(date)"
echo "Status: Running ckpt64 (3/3)" >> $STATUS_FILE
python RapidIn/MP_main.py --config_path configs/controlled_500/ckpt64_pretrain.json
echo "Completed: $(date)"
echo "Ckpt64 gradients:" $(ls results/controlled_experiment/grads/ckpt64/pretrain/*.pt 2>/dev/null | wc -l) "/500"

echo ""
echo "=============================================="
echo "All gradient computations complete!"
echo "Finished: $(date)"
echo "=============================================="

echo "Status: Complete" >> $STATUS_FILE

# Summary
echo ""
echo "=== SUMMARY ==="
echo "Baseline:" $(ls results/controlled_experiment/grads/baseline/pretrain/*.pt 2>/dev/null | wc -l) "/500"
echo "Ckpt32:" $(ls results/controlled_experiment/grads/ckpt32/pretrain/*.pt 2>/dev/null | wc -l) "/500"
echo "Ckpt64:" $(ls results/controlled_experiment/grads/ckpt64/pretrain/*.pt 2>/dev/null | wc -l) "/500"
