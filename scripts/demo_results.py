#!/usr/bin/env python3
"""Recompute paper comparisons from saved scores; no GPU or packages required."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ['baseline', 'ckpt32', 'ckpt64', 'ckpt96', 'ckpt128_lr2e-5', 'ckpt160_lr2e-5']


def load_scores(root, checkpoint, dataset):
    path = root / 'data-aggregate/outputs/influence_scores' / checkpoint / (dataset + '.json')
    obj = json.loads(path.read_text())
    if 'per_query' in obj:
        records = obj['per_query']
        scores = {r['test_idx']: r['max_influence'] for r in records}
        if len(scores) != len(records):
            raise ValueError(f'Duplicate query IDs in {path}')
    else:
        # Legacy lower-LR exports retain sample order, but omit explicit query IDs.
        scores = dict(enumerate(obj['per_sample']))
    if set(scores) != set(range(500)) or not all(math.isfinite(v) for v in scores.values()):
        raise ValueError(f'Expected 500 finite, aligned scores in {path}')
    return scores


def summarize(root=ROOT):
    rows = []
    for epoch, checkpoint in enumerate(CHECKPOINTS):
        ft = load_scores(root, checkpoint, 'finetune')
        pt = load_scores(root, checkpoint, 'iter4')
        wins = sum(pt[i] > ft[i] for i in ft)
        ties = sum(pt[i] == ft[i] for i in ft)
        rows.append(dict(epoch=epoch, checkpoint=checkpoint, n=500,
                         pt_wins=wins, pt_wins_pct=100*wins/500, ties=ties,
                         ft_mean=mean(ft.values()), pt_mean=mean(pt.values())))
    medical = load_scores(root, 'ckpt64', 'iter4')
    entertainment = load_scores(root, 'ckpt64', 'entertainment')
    med_mean, ent_mean = mean(medical.values()), mean(entertainment.values())
    comparison = json.loads((root / 'data-aggregate/outputs/analysis/bm25_accuracy_comparison/rapidin_vs_bm25_comparison.json').read_text())
    ft_pct = 100 - rows[2]['pt_wins_pct'] - 100*rows[2]['ties']/500
    return dict(training=rows, domain_comparison=dict(medical_mean=med_mean,
                entertainment_mean=ent_mean, relative_difference_pct=100*(med_mean/ent_mean-1)),
                method_comparison=dict(gradient_ft_wins_pct=ft_pct,
                bm25_ft_wins_pct=comparison['bm25']['ft_wins_pct'],
                difference_percentage_points=ft_pct-comparison['bm25']['ft_wins_pct']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional JSON output')
    args = parser.parse_args()
    result = summarize()
    print('Saved-score reproduction (not a new training run)')
    print('Epoch  Checkpoint           PT wins / 500  PT wins %')
    for r in result['training']:
        print(f"{r['epoch']:5d}  {r['checkpoint']:20s} {r['pt_wins']:5d} / 500    {r['pt_wins_pct']:5.1f}")
    print('Epochs 4-5 use the lower-learning-rate exports; their query alignment is positional.')
    print(f"Epoch 2 medical/control mean-score difference: {result['domain_comparison']['relative_difference_pct']:.1f}%")
    print(f"Gradient vs BM25 FT-selection difference: {result['method_comparison']['difference_percentage_points']:.1f} percentage points")
    print('FT/PT selection is not ground-truth attribution accuracy or a causal effect.')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
