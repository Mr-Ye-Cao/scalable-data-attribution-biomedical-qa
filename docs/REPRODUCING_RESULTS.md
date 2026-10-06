# Reproducing the saved results

## Laptop demo

Run `python3 scripts/demo_results.py --output results/demo-summary.json` from the repository root. Only Python's standard library is used. Input paths are resolved relative to the script, so invoking the script by absolute path also works from another directory.

The demo reads `data-aggregate/outputs/influence_scores/`, aligns the 500 test queries, and counts a PT win only when the maximum `iter4` medical-pretraining score is strictly larger than the maximum fine-tuning score. Ties are reported separately.

| Epoch | Score directory | PT wins / 500 | PT wins |
|---|---|---:|---:|
| 0 | baseline | 0 | 0.0% |
| 1 | ckpt32 | 9 | 1.8% |
| 2 | ckpt64 | 67 | 13.4% |
| 3 | ckpt96 | 83 | 16.6% |
| 4 | ckpt128_lr2e-5 | 124 | 24.8% |
| 5 | ckpt160_lr2e-5 | 142 | 28.4% |

Epochs 4–5 use the explicitly named lower-learning-rate exports. Their `per_sample` arrays omit query IDs; the demo assumes the original shared query order. The earlier exports have explicit `test_idx` values. The `ckpt128`/`ckpt160` directories without the suffix are a different run and must not be substituted into this series.

The domain comparison uses the saved epoch-2 `iter4` and `entertainment` mean maximum scores. Computing from full precision gives about 28.7%; the paper's 28.5% is reproduced by its rounded numbers 0.365 and 0.284. The BM25 comparison is taken from its archived analysis summary, while gradient FT wins are recomputed from per-query scores.

## Interpretation

These are gradient-similarity attribution scores, not causal interventions. A PT win is neither a proportion of all model knowledge nor a measure of answer accuracy. FT-selection disagreement with BM25 is not a ground-truth attribution accuracy test. The lower learning rate in later epochs is a changed experimental condition. These results concern this model, dataset, retrieval sample, and experiment; they do not establish a universal fine-tuning mechanism. Confidence intervals or multi-seed robustness are not supplied by this demo.

## Full experiments

Initialize only the code/data submodules you need:

```bash
git submodule update --init RapidIn Medical-LLM-Fine-tuning pubmedqa
```

Avoid initializing every submodule by default: some point to multi-billion-parameter model repositories. Review the external projects' requirements and licenses. The original training used a GH200-class environment; the laptop demo is not a replacement for a GPU rerun.

See [CODE_GUIDE.md](CODE_GUIDE.md) for training and attribution entry points. Historical configs refer to local paths such as `model_ckpts/checkpoint-64` and `data/rapidin_aligned`; the release does not include trained checkpoints or full projected-gradient caches. Recreate those inputs or adjust the configs before attempting the full pipeline. The archived command log is not a turnkey environment lockfile.

## Figure

With Matplotlib installed, run `python3 scripts/plot_summary.py` to recreate `assets/attribution-summary.png`. It reads the same saved scores through the demo module, rather than copying plotted constants.
