#!/bin/bash
# Run aligned gradient computation for 500 samples (test + finetune) for 3 models
#
# Usage: ./scripts/attribution/run_aligned_500.sh
#
# Total: 6 jobs (3 models x 2 data types)
# Estimated time: ~30-60 min per job = 3-6 hours total

set -e

cd /lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pubmed-gh200

export PYTHONPATH=".:./RapidIn:$PYTHONPATH"

echo "============================================================"
echo "Running Aligned Gradient Computation: 500 samples"
echo "Models: baseline, ckpt32, ckpt64"
echo "Data types: test, finetune"
echo "Started: $(date)"
echo "============================================================"

# Create output directories
mkdir -p results/rapidin_aligned_500/grads/baseline/{test,finetune}
mkdir -p results/rapidin_aligned_500/grads/ckpt32/{test,finetune}
mkdir -p results/rapidin_aligned_500/grads/ckpt64/{test,finetune}

# ============================================================
# Job 1: Baseline - Test (500 samples)
# ============================================================
echo ""
echo "[1/6] Baseline - Test (500 samples)"
echo "Started: $(date)"
python scripts/attribution/run_rapidin_original.py --config configs/aligned_500/baseline_test.json 2>&1 | tee results/rapidin_aligned_500/baseline_test.log
echo "Completed: $(date)"

# ============================================================
# Job 2: Baseline - Finetune (500 samples)
# ============================================================
echo ""
echo "[2/6] Baseline - Finetune (500 samples)"
echo "Started: $(date)"
python scripts/attribution/run_rapidin_original.py --config configs/aligned_500/baseline_finetune.json 2>&1 | tee results/rapidin_aligned_500/baseline_finetune.log
echo "Completed: $(date)"

# ============================================================
# Job 3: Ckpt32 - Test (500 samples)
# ============================================================
echo ""
echo "[3/6] Ckpt32 - Test (500 samples)"
echo "Started: $(date)"
python scripts/attribution/run_rapidin_original.py --config configs/aligned_500/ckpt32_test.json 2>&1 | tee results/rapidin_aligned_500/ckpt32_test.log
echo "Completed: $(date)"

# ============================================================
# Job 4: Ckpt32 - Finetune (500 samples)
# ============================================================
echo ""
echo "[4/6] Ckpt32 - Finetune (500 samples)"
echo "Started: $(date)"
python scripts/attribution/run_rapidin_original.py --config configs/aligned_500/ckpt32_finetune.json 2>&1 | tee results/rapidin_aligned_500/ckpt32_finetune.log
echo "Completed: $(date)"

# ============================================================
# Job 5: Ckpt64 - Test (500 samples)
# ============================================================
echo ""
echo "[5/6] Ckpt64 - Test (500 samples)"
echo "Started: $(date)"
python scripts/attribution/run_rapidin_original.py --config configs/aligned_500/ckpt64_test.json 2>&1 | tee results/rapidin_aligned_500/ckpt64_test.log
echo "Completed: $(date)"

# ============================================================
# Job 6: Ckpt64 - Finetune (500 samples)
# ============================================================
echo ""
echo "[6/6] Ckpt64 - Finetune (500 samples)"
echo "Started: $(date)"
python scripts/attribution/run_rapidin_original.py --config configs/aligned_500/ckpt64_finetune.json 2>&1 | tee results/rapidin_aligned_500/ckpt64_finetune.log
echo "Completed: $(date)"

echo ""
echo "============================================================"
echo "ALL GRADIENT JOBS COMPLETED!"
echo "Finished: $(date)"
echo "============================================================"
echo ""
echo "Gradient directories:"
echo "  - results/rapidin_aligned_500/grads/baseline/{test,finetune}/"
echo "  - results/rapidin_aligned_500/grads/ckpt32/{test,finetune}/"
echo "  - results/rapidin_aligned_500/grads/ckpt64/{test,finetune}/"
echo ""
echo "To verify counts:"
echo "  find results/rapidin_aligned_500/grads -name '*.pt' | wc -l"
echo "  (Expected: 3000 total = 3 models x 2 types x 500 samples)"
