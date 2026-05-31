# Review Response Roadmap

The reviews are broadly positive about the research question and storage-efficiency story, but they converge on the same empirical concerns. This roadmap turns those concerns into concrete repo tasks.

## Reviewer Concerns

| Concern | Review Signal | Current Evidence | What To Add |
| --- | --- | --- | --- |
| Approximation validity | Identity Hessian, gradient inner products, OPORP compression, and pseudo-label/self-influence need validation. | LARK/RapidIn is compared with BM25 and storage estimates, but there is no small-scale faithful influence check. | Add a small HVP/CG or restricted-parameter influence validation on a subset, and report rank correlation with LARK. |
| Robustness | Reviewers ask for seeds, projection dimensions, projection seeds, confidence intervals, and variance estimates. | Current results use 500 samples and fixed projection settings. | Run `k` sensitivity, OPORP seed sensitivity, and bootstrap confidence intervals over test queries. |
| Format vs content confounding | Reviewers note that QA formatting may dominate influence. | Existing controls include no-format, formatted medical, and formatted entertainment sources. | Promote these controls more clearly, add paraphrase/template variants, and report content-only deltas separately from template deltas. |
| Baseline strength | BM25 is considered too weak to support mechanistic claims. | Current comparison shows LARK `86.6%` vs BM25 `40.4%` FT wins. | Add dense retrieval, simple gradient dot product without OPORP, TracIn on a tiny subset, or Fisher/diagonal approximations where feasible. |
| Gradient magnitude artifact | One review asks whether "awakened" points are just large-gradient points. | The paper says cosine similarity normalizes projected gradients, but does not show a dedicated norm ablation. | Report influence with and without gradient-norm normalization, plus top-attributed sample norm distributions. |
| Claim strength | Reviewers object to strong causal language like "unlocks pretrained knowledge." | The paper frames the phenomenon as "knowledge awakening." | Use more cautious wording unless validation is added: "increasing pretrained-data attribution during fine-tuning" rather than a causal mechanistic claim. |
| Numeric consistency | One review flags `54,000x` vs `32,768x` compression. | Table III uses `130 MB`; Figure 5 uses `223 MB`. | Reconcile storage calculations and state whether numbers use one source set, 500 samples, or measured RapidIn files. |

## Highest-Value Next Experiments

1. **Projection robustness.** Re-run attribution for `k in {8192, 16384, 32768, 65536}` and at least 3 OPORP seeds. Report PT-wins mean/std and rank stability.
2. **Bootstrap intervals.** Use existing per-query scores to bootstrap confidence intervals for mean max influence, FT/PT ratio, PT wins, and medical-vs-entertainment lift.
3. **Norm artifact ablation.** Compare cosine-normalized influence with raw dot product and gradient-norm-only ranking.
4. **Small faithful influence sanity check.** On a small subset and/or final-layer-only parameter slice, compare LARK rankings to HVP/CG influence or a leave-one-out style approximation.
5. **Stronger retrieval baseline.** Add dense embedding retrieval over the same candidate pools, then compare source attribution behavior with BM25 and LARK.
6. **Template robustness.** Add multiple QA templates/paraphrases and a content-only control that preserves medical content while varying prompt wrapper.

## Repo Starting Points

| Task | Start From |
| --- | --- |
| Reproduce current analysis | `paper-writing` or `organized-results-display` |
| Add robustness scripts | `iterate-gh200-embedding-attribution` |
| Recover OLMo-3 fine-tuning commands | `iterate-gh200-finetune-olmo3` |
| Recover table/report variants | `iterate-gh200-embedding-attribution-analysis-plots` |

## Suggested Camera-Ready Framing

Use cautious language until the validation experiments land:

> We observe that attribution to retrieved medical pretraining documents increases during fine-tuning, consistent with the model relying more on pretrained domain knowledge after task adaptation.

Avoid stronger phrasing unless backed by the new checks:

> Fine-tuning unlocks pretrained knowledge.

The stronger phrase can still be used as an intuition or hypothesis, but the measurable claim should be the attribution trend.
