# Scalable Data Attribution for Biomedical QA

This branch is an organized results view for the IEEE ICHI 2026 submission:

**Scalable Data Attribution Reveals Fine-Tuning Unlocks LLM Pretrained Knowledge in Biomedical Question Answering**

The project studies whether a biomedical QA model's predictions are driven more by fine-tuning examples or by pretrained medical knowledge. The experiments use OLMo-3-7B-Instruct fine-tuned on PubMedQA and compare gradient-based attribution against lexical retrieval baselines.

## Key Takeaways

| Question | Current Evidence |
| --- | --- |
| Can attribution separate medical from non-medical data? | Medical pretraining text has higher influence than entertainment controls under the same formatting. At epoch 2, medical formatted PT mean max influence is `0.365` vs entertainment formatted PT `0.284`, a `28.5%` lift. |
| Does pretraining influence change during fine-tuning? | PT wins rise from `0.0%` at baseline to `28.4%` by epoch 5, suggesting increasing use of pretrained medical knowledge as training progresses. |
| Does gradient attribution beat lexical matching? | LARK/RapidIn FT wins are `86.6%` at epoch 2, while BM25 FT wins are `40.4%`, a `46.2` point gap. |
| Is full-gradient attribution feasible? | Full-gradient TracIn for 500 samples is estimated at about `7.3 TB`; LARK/RapidIn projected storage is about `223 MB` for the same sample count. |

## Result Figures

| Result | Figure |
| --- | --- |
| Domain relevance with formatted data | ![Formatted data comparison](ICHI_Data_Valuation/figures/fig1a_formatted_data.png) |
| Domain relevance without QA formatting | ![Raw data comparison](ICHI_Data_Valuation/figures/fig1b_raw_data.png) |
| Fine-tune vs pretrain influence | ![FT vs PT influence](ICHI_Data_Valuation/figures/fig2a_ft_vs_pt_influence.png) |
| PT wins over training | ![PT wins percentage](ICHI_Data_Valuation/figures/fig2b_pt_wins_pct.png) |
| Storage efficiency | ![Storage efficiency](ICHI_Data_Valuation/figures/fig5a_storage_efficiency.png) |
| LARK vs BM25 attribution | ![Attribution accuracy](ICHI_Data_Valuation/figures/fig5b_attribution_accuracy.png) |

## Repository Layout

| Path | Purpose |
| --- | --- |
| `ICHI_Data_Valuation/` | LaTeX paper source, generated PDF, paper-ready figures, bibliography, and IEEE template assets. |
| `scripts/` | Data preparation, evaluation, attribution, BM25, TracIn-memory, and analysis scripts. |
| `configs/` | RapidIn/LARK experiment configs for baseline, checkpoints, controls, learning-rate variants, and validation runs. |
| `data-aggregate/` | Compact input samples, influence-score summaries, analysis JSON, and figure-generation scripts used by the paper. |
| `data-aggregate/DATA.md` | Data dictionary for the aggregate folder, including checkpoint names, source definitions, and JSON schemas. |
| `CMD.md` | Command log for data download, model evaluation, fine-tuning, RapidIn/LARK attribution, and monitoring long jobs. |
| `Medical-LLM-Fine-tuning/` | Submodule pointing to the fine-tuning framework used for OLMo/PubMedQA training. |
| `docs/BRANCH_MAP.md` | Guide to the remote branches and which one to use for each kind of follow-up. |
| `docs/CODE_GUIDE.md` | Quick guide to the training and attribution code paths. |
| `docs/REVIEW_RESPONSE_ROADMAP.md` | Reviewer-driven plan for strengthening the empirical evidence and tightening claims. |

## Code Entry Points

Initialize external code/model/data references:

```bash
git submodule update --init --recursive
```

Common entry points:

```bash
# Download PubMedQA
python scripts/data_prep/download_pubmedqa.py

# Evaluate base model outputs
python scripts/eval/eval_base_model.py

# Fine-tune via the Medical-LLM-Fine-tuning submodule
cd Medical-LLM-Fine-tuning
python train.py --model_type olmo3-7b-instruct --subset pqa_labeled --epochs 3 --full_finetune

# Run RapidIn/LARK attribution with a saved config
python scripts/attribution/run_rapidin_original.py --config configs/aligned_500/ckpt64_finetune.json
```

See `docs/CODE_GUIDE.md` and `CMD.md` for the fuller pipeline.

## Reproducing Display Figures

The display figures are already checked in. To regenerate them from the aggregate result constants:

```bash
python data-aggregate/generate_figures.py
```

The script writes into `ICHI_Data_Valuation/figures/`.

## Paper And Reviews

The submitted paper is at `ICHI_Data_Valuation/main.pdf`. The reviews mostly ask for stronger validation of the attribution approximation, robustness across seeds/projection dimensions, clearer separation of content from format, stronger baselines, and toned-down causal language. See `docs/REVIEW_RESPONSE_ROADMAP.md` for a concrete follow-up plan.
