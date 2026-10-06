# Code Guide

Start with [the no-GPU saved-score demo](REPRODUCING_RESULTS.md). Full pipeline reruns also require external dependencies, model weights, prepared data, and trained checkpoints; they have not been revalidated for this release.

## External Dependencies

The following are included as submodule references rather than copied into this repo:

| Path | Purpose |
| --- | --- |
| `Medical-LLM-Fine-tuning/` | Fine-tuning framework used for PubMedQA model training. |
| `RapidIn/` | RapidIn/LARK gradient attribution implementation. |
| `pubmedqa/` | Official PubMedQA dataset/evaluation repository. |
| `OLMo-2-0425-1B/` | OLMo-2 model reference. |
| `OLMo-3-7B-Instruct/` | OLMo-3 instruct model reference used by the paper. |
| `OLMo-3-1025-7B/` | OLMo-3 base model reference. |

Initialize them with:

```bash
git submodule update --init RapidIn Medical-LLM-Fine-tuning pubmedqa
```

## Training And Evaluation

| Task | Entry Point |
| --- | --- |
| PubMedQA download | `scripts/data_prep/download_pubmedqa.py` |
| OLMo/PubMedQA fine-tuning | `Medical-LLM-Fine-tuning/train.py` |
| Base model evaluation | `scripts/eval/eval_base_model.py` |
| Gradient benchmark | `scripts/eval/benchmark_olmo2_gradient.py` |
| Embedding extraction | `scripts/eval/extract_embeddings_vllm.py` |

The most complete command log is `CMD.md`.

## Attribution And Controls

| Task | Entry Point |
| --- | --- |
| RapidIn/LARK attribution | `scripts/attribution/run_rapidin_original.py` |
| Batch gradient runs | `scripts/attribution/run_all_gradients.sh` and related shell scripts |
| Fine-tune vs pretrain comparison | `scripts/attribution/compare_pretrain_vs_finetune.py` |
| Format-control experiments | `scripts/attribution/controlled_experiment/` |
| BM25 baseline | `scripts/attribution/goal3/bm25_baseline.py` |
| RapidIn vs BM25 comparison | `scripts/attribution/goal3/compare_rapidin_bm25.py` |
| TracIn memory estimate | `scripts/attribution/goal3/tracin_memory_estimation.py` |
| Analysis and plotting | `scripts/attribution/validation/` and `scripts/attribution/visualize_influence.py` |

## Configs

Experiment configs are organized under `configs/`:

| Directory | Purpose |
| --- | --- |
| `configs/aligned_500/` | Main 500-sample aligned fine-tune/test attribution configs. |
| `configs/controlled_500/` | Format-control and template-control pretraining configs. |
| `configs/new_epochs/` | Later checkpoint configs. |
| `configs/lr2e-5/` | Lower learning-rate checkpoint configs. |
| `configs/entertainment_pretrain/` | Non-medical entertainment control configs. |
| `configs/validation/comprehensive/` | Comprehensive validation configs. |

## Result Data

Compact result summaries live in `data-aggregate/`. See `data-aggregate/DATA.md` for the data dictionary and schema descriptions.
