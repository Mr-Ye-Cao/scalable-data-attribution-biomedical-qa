import matplotlib.pyplot as plt
import numpy as np
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, '..', 'ICHI_Data_Valuation', 'figures')

# Set style with LARGE fonts for paper readability
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'

# Updated data with lower LR for epochs 4-5
epochs = [0, 1, 2, 3, 4, 5]
epoch_labels = ['Epoch 0', 'Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4*', 'Epoch 5*']

# Influence scores (updated with lower LR for last 2)
finetune_fmt = [0.4815, 0.4734, 0.3862, 0.3436, 0.2623, 0.2575]
pt_iter1_fmt = [0.4019, 0.3968, 0.3267, 0.2989, 0.2286, 0.2264]
pt_iter4_fmt = [0.3946, 0.4310, 0.3651, 0.3268, 0.2537, 0.2501]
pt_noformat = [0.1086, 0.1340, 0.1173, 0.1103, 0.0278, 0.0268]
ent_fmt = [0.3839, 0.3200, 0.2838, 0.2734, 0.2096, 0.2081]
ent_noformat = [0.1035, 0.0969, 0.0951, 0.0911, 0.0071, 0.0067]

# PT wins data (updated)
pt_wins_iter4 = [0, 9, 67, 83, 124, 142]
pt_wins_pct = [x/5 for x in pt_wins_iter4]  # Convert to percentage

# Colors
colors = {
    'finetune': '#2E86AB',      # Blue
    'pt_fmt': '#A23B72',        # Magenta
    'pt_raw': '#F18F01',        # Orange
    'ent_fmt': '#C73E1D',       # Red
    'ent_raw': '#6B717E',       # Gray
}

# ============================================================
# Figure 1A: Formatted data comparison (single column)
# ============================================================
fig1a, ax1 = plt.subplots(figsize=(8, 6))
x = np.arange(len(epochs))
width = 0.25

bars1 = ax1.bar(x - width, finetune_fmt, width, label='Fine-tune (Formatted)', color=colors['finetune'], edgecolor='white')
bars2 = ax1.bar(x, pt_iter1_fmt, width, label='Medical PT (Formatted)', color=colors['pt_fmt'], edgecolor='white')
bars3 = ax1.bar(x + width, ent_fmt, width, label='Entertainment (Formatted)', color=colors['ent_fmt'], edgecolor='white')

ax1.set_xlabel('Training Epoch')
ax1.set_ylabel('Mean Max Influence Score')
ax1.set_xticks(x)
ax1.set_xticklabels(epoch_labels)
ax1.legend(loc='upper right', fontsize=14)
ax1.set_ylim(0, 0.55)

# Add significance markers
for i in range(len(epochs)):
    ax1.annotate('***', xy=(x[i] - width/2, max(finetune_fmt[i], ent_fmt[i]) + 0.02),
                 ha='center', fontsize=12, color='black')

fig1a.text(0.5, 0.02, '* Epochs 4-5 use lower learning rate (2e-5 instead of 2e-4)',
          ha='center', fontsize=12, style='italic', color='gray')

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(os.path.join(output_dir, 'fig1a_formatted_data.png'), dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Figure 1A saved: Formatted Data Comparison")

# ============================================================
# Figure 1B: Raw data comparison (single column)
# ============================================================
fig1b, ax2 = plt.subplots(figsize=(8, 6))
bars4 = ax2.bar(x - width/2, pt_noformat, width, label='Medical PT (Raw)', color=colors['pt_raw'], edgecolor='white')
bars5 = ax2.bar(x + width/2, ent_noformat, width, label='Entertainment (Raw)', color=colors['ent_raw'], edgecolor='white')

ax2.set_xlabel('Training Epoch')
ax2.set_ylabel('Mean Max Influence Score')
ax2.set_xticks(x)
ax2.set_xticklabels(epoch_labels)
ax2.legend(loc='upper right', fontsize=14)
ax2.set_ylim(0, 0.16)

fig1b.text(0.5, 0.02, '* Epochs 4-5 use lower learning rate (2e-5 instead of 2e-4)',
          ha='center', fontsize=12, style='italic', color='gray')

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(os.path.join(output_dir, 'fig1b_raw_data.png'), dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Figure 1B saved: Raw Data Comparison")

# ============================================================
# Figure 2A: FT vs PT Mean Influence (single column)
# ============================================================
fig2a, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(epochs, finetune_fmt, 'o-', linewidth=2.5, markersize=10,
         label='Fine-tune', color=colors['finetune'])
ax1.plot(epochs, pt_iter4_fmt, 's-', linewidth=2.5, markersize=10,
         label='Pretrain (iter4)', color=colors['pt_fmt'])

ax1.axvspan(3.5, 5.5, alpha=0.15, color='yellow', label='Lower LR (2e-5)')

ax1.set_xlabel('Training Epoch')
ax1.set_ylabel('Mean Max Influence Score')
ax1.set_xticks(epochs)
ax1.set_xticklabels(epoch_labels)
ax1.legend(loc='upper right', fontsize=14)
ax1.set_ylim(0.2, 0.55)

ft_pt_ratios = [f/p for f, p in zip(finetune_fmt, pt_iter4_fmt)]
for i, ratio in enumerate(ft_pt_ratios):
    ax1.annotate(f'{ratio:.2f}x', xy=(epochs[i], finetune_fmt[i] + 0.02),
                 ha='center', fontsize=14, color='gray')

fig2a.text(0.5, 0.02, '* Epochs 4-5 use lower learning rate (2e-5)',
          ha='center', fontsize=12, style='italic', color='gray')

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(os.path.join(output_dir, 'fig2a_ft_vs_pt_influence.png'), dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Figure 2A saved: FT vs PT Mean Influence")

# ============================================================
# Figure 2B: PT Wins Percentage (single column)
# ============================================================
fig2b, ax2 = plt.subplots(figsize=(8, 6))
bars = ax2.bar(epochs, pt_wins_pct, color=colors['pt_fmt'], edgecolor='white', width=0.6)

bars[4].set_color('#7B2D8E')
bars[5].set_color('#7B2D8E')

ax2.set_xlabel('Training Epoch')
ax2.set_ylabel('PT Wins (%)')
ax2.set_xticks(epochs)
ax2.set_xticklabels(epoch_labels)
ax2.set_ylim(0, 35)

for i, (bar, pct) in enumerate(zip(bars, pt_wins_pct)):
    ax2.annotate(f'{pct:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8),
                 ha='center', fontsize=14, fontweight='bold')

fig2b.text(0.5, 0.02, '* Epochs 4-5 use lower learning rate (2e-5) - shows dramatic increase in PT importance',
          ha='center', fontsize=12, style='italic', color='gray')

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(os.path.join(output_dir, 'fig2b_pt_wins_pct.png'), dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Figure 2B saved: PT Wins Percentage")

# ============================================================
# Figure 3A: PT Wins - Learning Rate Comparison (single column)
# ============================================================
fig3a, ax1 = plt.subplots(figsize=(8, 6))

lr_epochs = ['Epoch 4', 'Epoch 5']
original_lr_pt_wins = [7.4, 8.8]
lower_lr_pt_wins = [24.8, 28.4]
original_lr_acc = [72.2, 70.8]
lower_lr_acc = [74.2, 74.0]

x = np.arange(len(lr_epochs))
width = 0.35

bars1 = ax1.bar(x - width/2, original_lr_pt_wins, width, label='Original LR (2e-4)',
                color='#C0C0C0', edgecolor='white')
bars2 = ax1.bar(x + width/2, lower_lr_pt_wins, width, label='Lower LR (2e-5)',
                color=colors['pt_fmt'], edgecolor='white')

ax1.set_xlabel('Training Epoch')
ax1.set_ylabel('PT Wins (%)')
ax1.set_xticks(x)
ax1.set_xticklabels(lr_epochs)
ax1.legend(loc='upper left', fontsize=14)
ax1.set_ylim(0, 35)

for bars in [bars1, bars2]:
    for bar in bars:
        ax1.annotate(f'{bar.get_height():.1f}%',
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8),
                     ha='center', fontsize=14, fontweight='bold')

for i in range(2):
    ax1.annotate('', xy=(x[i] + width/2, lower_lr_pt_wins[i] - 1),
                 xytext=(x[i] - width/2, original_lr_pt_wins[i] + 1),
                 arrowprops=dict(arrowstyle='->', color='green', lw=2))
    increase = lower_lr_pt_wins[i] / original_lr_pt_wins[i]
    ax1.text(x[i], (original_lr_pt_wins[i] + lower_lr_pt_wins[i])/2,
             f'{increase:.1f}x', ha='center', fontsize=16, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig3a_lr_pt_wins.png'), dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Figure 3A saved: LR Comparison - PT Wins")

# ============================================================
# Figure 3B: Test Accuracy - Learning Rate Comparison (single column)
# ============================================================
fig3b, ax2 = plt.subplots(figsize=(8, 6))
bars3 = ax2.bar(x - width/2, original_lr_acc, width, label='Original LR (2e-4)',
                color='#C0C0C0', edgecolor='white')
bars4 = ax2.bar(x + width/2, lower_lr_acc, width, label='Lower LR (2e-5)',
                color=colors['finetune'], edgecolor='white')

ax2.set_xlabel('Training Epoch')
ax2.set_ylabel('Test Accuracy (%)')
ax2.set_xticks(x)
ax2.set_xticklabels(lr_epochs)
ax2.legend(loc='lower right', fontsize=14)
ax2.set_ylim(68, 76)

for bars in [bars3, bars4]:
    for bar in bars:
        ax2.annotate(f'{bar.get_height():.1f}%',
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3),
                     ha='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig3b_lr_accuracy.png'), dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Figure 3B saved: LR Comparison - Accuracy")

# ============================================================
# Figure 4: PT Importance Over Training (single column, unchanged)
# ============================================================
fig4, ax = plt.subplots(figsize=(8, 6))

ax.plot(epochs, pt_wins_pct, 'o-', linewidth=3, markersize=14,
        color=colors['pt_fmt'], label='PT Wins %')

ax.fill_between(epochs, pt_wins_pct, alpha=0.3, color=colors['pt_fmt'])

ax.axvspan(3.5, 5.5, alpha=0.2, color='yellow')
ax.text(4.5, 32, 'Lower LR\n(2e-5)', ha='center', fontsize=16,
        style='italic', color='#666666')

ax.annotate('Format learning\nphase', xy=(0.5, 2), fontsize=14,
            ha='center', color='gray')
ax.annotate('Knowledge\nutilization phase', xy=(3, 20), fontsize=14,
            ha='center', color='gray')

for i, pct in enumerate(pt_wins_pct):
    ax.annotate(f'{pct:.1f}%', xy=(epochs[i], pct + 1.5),
                ha='center', fontsize=16, fontweight='bold')

ax.set_xlabel('Training Epoch', fontsize=20)
ax.set_ylabel('Queries Where Pretrain > Finetune (%)', fontsize=20)
# Title removed (moved to caption)
ax.set_xticks(epochs)
ax.set_xticklabels(epoch_labels, fontsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.set_ylim(0, 35)
ax.set_xlim(-0.5, 5.5)

ax.axhline(y=28.4, color='gray', linestyle='--', alpha=0.5)
ax.text(5.3, 29, '28.4%', fontsize=14, color='gray')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig4_pt_importance_trend.png'), dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Figure 4 saved: PT Importance Trend")

# ============================================================
# Figure 5A: Storage Efficiency (single column)
# ============================================================
fig5a, ax1 = plt.subplots(figsize=(8, 6))
methods = ['TracIn', 'LARK']
storage = [7300, 0.223]  # GB for 500 samples (TracIn=7.3TB, LARK=223MB)

bars = ax1.bar(methods, storage, color=[colors['ent_fmt'], colors['finetune']],
               edgecolor='white', width=0.5)
ax1.set_yscale('log')
ax1.set_ylabel('Storage for 500 Samples (GB, log scale)')
ax1.set_ylim(0.1, 20000)

ax1.annotate('7.3 TB', xy=(0, 7300), ha='center', va='bottom', fontsize=18, fontweight='bold')
ax1.annotate('223 MB', xy=(1, 0.223), ha='center', va='bottom', fontsize=18, fontweight='bold')

ax1.annotate('', xy=(1, 1), xytext=(0, 1000),
             arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax1.text(0.5, 30, '32,768x\nsmaller!', ha='center', fontsize=16,
         color='green', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig5a_storage_efficiency.png'), dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Figure 5A saved: Storage Efficiency")

# ============================================================
# Figure 5B: Attribution Accuracy (single column)
# ============================================================
fig5b, ax2 = plt.subplots(figsize=(8, 6))
methods2 = ['BM25\n(Lexical)', 'LARK\n(Gradient)']
accuracy = [40.4, 86.6]

bars2 = ax2.bar(methods2, accuracy, color=[colors['ent_raw'], colors['finetune']],
                edgecolor='white', width=0.5)
ax2.set_ylabel('FT > PT Wins (%)')
ax2.set_ylim(0, 100)

for bar, acc in zip(bars2, accuracy):
    ax2.annotate(f'{acc}%', xy=(bar.get_x() + bar.get_width()/2, acc + 2),
                 ha='center', fontsize=18, fontweight='bold')

ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
ax2.text(1.3, 52, 'Random\n(50%)', fontsize=14, color='gray')

ax2.annotate('', xy=(1, 86.6 - 3), xytext=(0, 40.4 + 3),
             arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax2.text(0.5, 63, '+46.2%', ha='center', fontsize=16,
         color='green', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig5b_attribution_accuracy.png'), dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Figure 5B saved: Attribution Accuracy")

print("\nAll figures generated successfully!")
print(f"Location: {output_dir}")
