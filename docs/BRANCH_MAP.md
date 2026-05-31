# Branch Map

This repo has a linear research history with several branch snapshots. `main` is only the initial placeholder, so the useful work lives on the experiment and writing branches.

## Recommended Branch Roles

| Branch | Purpose | Best Use |
| --- | --- | --- |
| `organized-results-display` | Clean presentation branch created from `paper-writing`. | Use this for GitHub-facing result display, paper links, branch guide, and reviewer-response planning. |
| `paper-writing` | Final ICHI paper assets plus `data-aggregate` results and paper figures. | Best source for the submitted manuscript, final result figures, compact aggregate data, and paper-ready narrative. |
| `iterate-gh200-embedding-attribution` | Most complete experiment/code branch. Includes configs, data prep, attribution scripts, validation scripts, BM25 comparison, TracIn memory estimates, and aggregate outputs. | Best source when reproducing or extending experiments. Start here for new robustness checks, baselines, projection-seed studies, or reviewer-requested analyses. |
| `iterate-gh200-embedding-attribution-analysis-plots` | Analysis/plotting snapshot with report-style tables, top-k analyses, and figure variants. | Useful for recovering analysis reports or table-generation ideas that were not carried into `paper-writing`. |
| `analyze-data` | Earlier OLMo-3 attribution pipeline with compact scripts and first organized data notes. | Good for understanding the first clean version of the RapidIn/OLMo workflow. Less complete than the later GH200 attribution branch. |
| `iterate-gh200-finetune-olmo3` | GH200 full fine-tuning branch. Records OLMo-3 full fine-tuning, official eval flow, checkpoint saving, and cleanup of redundant eval scripts. | Use this to recover training/evaluation commands and checkpoint provenance. |
| `iterate-rtx5090-embedding-attribution` | Early local/RTX5090 attribution exploration. Adds embedding attribution, eval scripts, and first RapidIn OLMo adaptation. | Historical reference for early prototype code and local hardware experiments. Not the best base for final results. |
| `under-goal-1` | Early baseline work: data download/process, gradient computation, initial RapidIn CPU offload, fine-tuning, and baseline eval. | Use only for tracing the initial implementation lineage. |
| `main` | Initial commit only. | Do not use as a source for results. |

## Suggested Workflow

1. Use `organized-results-display` for clean public-facing material.
2. Use `paper-writing` when checking what was actually submitted.
3. Use `iterate-gh200-embedding-attribution` for code-backed follow-up experiments.
4. Cherry-pick or copy specific analysis docs from `iterate-gh200-embedding-attribution-analysis-plots` if a future display page needs more tables or top-k visualizations.
5. Refer to `iterate-gh200-finetune-olmo3` when explaining how the OLMo-3 checkpoints were produced.

## Why This Branch Starts From `paper-writing`

The display branch should be easy to browse. `paper-writing` already has the paper, figures, aggregate input samples, and compact result summaries without the large scratch-work surface of the full experiment branch. The more complete experiment code remains available on `iterate-gh200-embedding-attribution` and can be merged selectively when a reproducibility-focused branch is needed.
