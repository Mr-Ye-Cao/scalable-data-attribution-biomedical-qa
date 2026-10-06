#!/usr/bin/env python3
"""Create a shareable figure from the archived per-query score comparisons."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from demo_results import summarize

ROOT = Path(__file__).resolve().parents[1]
data = summarize()['training']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13,
                     'text.color': '#172B38', 'axes.labelcolor': '#172B38',
                     'xtick.color': '#172B38', 'ytick.color': '#172B38'})
fig = plt.figure(figsize=(12, 7), facecolor='white')
fig.text(.08, .92, 'Which examples receive the higher attribution score?', fontsize=22, weight='bold')
fig.text(.08, .864, 'OLMo-3-7B-Instruct on PubMedQA · 500 test queries · saved experiment results', fontsize=13)
ax = fig.add_axes([.09, .29, .81, .48])
x = [r['epoch'] for r in data]; y = [r['pt_wins_pct'] for r in data]
ax.plot(x[:4], y[:4], '-o', color='#176EAD', lw=2.7, ms=8)
ax.plot(x[3:], y[3:], '--o', color='#176EAD', lw=2.7, ms=8)
for r in data:
    ax.annotate(f"{r['pt_wins_pct']:.1f}%", (r['epoch'], r['pt_wins_pct']), xytext=(0, 13), textcoords='offset points', ha='center', fontsize=14, weight='bold')
ax.set(xlim=(-.25, 5.25), ylim=(0, 36), xticks=x, yticks=[0,10,20,30], xlabel='Fine-tuning epoch', ylabel='Queries where PT score > FT score')
ax.yaxis.set_major_formatter(PercentFormatter(100))
ax.grid(axis='y', color='#DFE5E8', lw=.8)
ax.set_axisbelow(True)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
for sp in ['left', 'bottom']: ax.spines[sp].set_color('#93A1AA')
fig.text(.09, .18, 'PT = sampled medical pretraining examples; FT = fine-tuning examples.', fontsize=11)
fig.text(.09, .142, 'Dashed segment: lower learning rate in epochs 4–5. Scores are associations, not causal proof.', fontsize=11)
fig.text(.09, .104, 'One experimental series; no confidence intervals shown. Not the percentage of knowledge from pretraining.', fontsize=10.5)
fig.text(.09, .047, 'Ye Cao & Zhaozhuo Xu · IEEE ICHI 2026 · doi.org/10.1109/ICHI69079.2026.00228', fontsize=11, color='#47606F')
(ROOT / 'assets').mkdir(exist_ok=True)
fig.savefig(ROOT / 'assets/attribution-summary.png', dpi=180, facecolor='white')
plt.close(fig)
