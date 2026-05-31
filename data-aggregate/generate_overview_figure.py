import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Polygon, PathPatch
from matplotlib.path import Path
import numpy as np
import os

def draw_robot(ax, x, y, size=0.35):
    """Draw a cute robot icon"""
    body = FancyBboxPatch((x - size/2, y - size/2), size, size,
                           boxstyle="round,pad=0.01,rounding_size=0.06",
                           facecolor='#4CAF50', edgecolor='#4CAF50', linewidth=1)
    ax.add_patch(body)
    eye_r = size * 0.11
    ax.add_patch(Circle((x - size*0.2, y + size*0.08), eye_r, facecolor='white', edgecolor='none'))
    ax.add_patch(Circle((x + size*0.2, y + size*0.08), eye_r, facecolor='white', edgecolor='none'))
    # Antenna
    ax.plot([x, x], [y + size/2, y + size/2 + size*0.25], color='#4CAF50', linewidth=1.5)
    ax.add_patch(Circle((x, y + size/2 + size*0.25), size*0.07, facecolor='#4CAF50', edgecolor='#4CAF50', linewidth=1))

def draw_person(ax, x, y, size=0.35, color='#1976D2'):
    """Draw a simple person icon"""
    ax.add_patch(Circle((x, y + size*0.4), size*0.22, facecolor=color, edgecolor='none'))
    body_pts = [(x - size*0.32, y - size*0.25), (x + size*0.32, y - size*0.25),
                (x + size*0.22, y + size*0.15), (x - size*0.22, y + size*0.15)]
    ax.add_patch(Polygon(body_pts, closed=True, facecolor=color, edgecolor='none'))

def draw_thought_cloud(ax, x, y, width, height):
    """Draw a thought bubble with cloud-like appearance"""
    # Main cloud body using multiple overlapping ellipses
    from matplotlib.patches import Ellipse as Ell

    # Main body
    main = FancyBboxPatch((x, y), width, height,
                          boxstyle="round,pad=0.02,rounding_size=0.25",
                          facecolor='white', edgecolor='#333333', linewidth=1.2)
    ax.add_patch(main)

    # Small bubbles leading to person
    ax.add_patch(Circle((x + 0.15, y - 0.12), 0.08, facecolor='white', edgecolor='#333333', linewidth=1.2))
    ax.add_patch(Circle((x + 0.05, y - 0.28), 0.05, facecolor='white', edgecolor='#333333', linewidth=1.2))

# Create figure
fig, ax = plt.subplots(figsize=(13, 9.5)) # Increased height significantly
ax.set_xlim(0, 10)
ax.set_ylim(0, 8.5)
ax.axis('off')

# Colors
blue_light = '#E3F2FD'
blue_dark = '#1976D2'
green_light = '#E8F5E9'
green_dark = '#4CAF50'
gray_light = '#F8F9FA'
gray_border = '#E0E0E0'

# Base font sizes
fs_small = 14
fs_med = 16
fs_large = 19
fs_xl = 24

# ============================================================
# TOP LEFT: User question bubble
# ============================================================
# Person icon
draw_person(ax, 0.6, 7.8, size=0.45, color=blue_dark)

# Question bubble (simple rounded rectangle)
q_box = FancyBboxPatch((1.1, 7.4), 4.4, 0.8, boxstyle="round,pad=0.02,rounding_size=0.15",
                        facecolor='white', edgecolor=blue_dark, linewidth=2.0)
ax.add_patch(q_box)
ax.text(3.3, 7.8, 'Is endosonography valuable\nin dyschesia?',
        fontsize=fs_large, ha='center', va='center', color='#333333')

# Response bubble
r_box = FancyBboxPatch((1.3, 6.3), 4.6, 0.9, boxstyle="round,pad=0.02,rounding_size=0.15",
                        facecolor='white', edgecolor=green_dark, linewidth=2.0)
ax.add_patch(r_box)
ax.text(3.6, 6.75, 'Yes. Endosonography demonstrated\nincomplete relaxation...',
        fontsize=fs_med, ha='center', va='center', color='#333333')

# Robot icon next to response
draw_robot(ax, 6.4, 6.75, size=0.5)

# Label below chat
ax.text(3.6, 5.9, 'Test Generation', fontsize=fs_large, ha='center', va='center', fontweight='bold', color='#333333')
ax.text(5.1, 5.9, 't', fontsize=fs_xl, ha='left', va='center', style='italic', fontweight='bold', color='#333333')

# ============================================================
# TOP RIGHT: Training Dataset
# ============================================================
ellipse = Ellipse((8.4, 7.1), 3.2, 2.4, facecolor='none', edgecolor='#333333',
                   linewidth=2.0, linestyle='--')
ax.add_patch(ellipse)

# Data points - slightly larger, strictly within ellipse
np.random.seed(42)  # Keep seed for reproducibility
# Ellipse center (8.4, 7.1), width=3.2 (rx=1.6), height=2.4 (ry=1.2)
# We generate points in a smaller box to stay safely inside
for i in range(12):
    # Generate points closer to center to avoiding overlapping the border
    # Range x: [7.6, 9.2], Range y: [6.5, 7.7]
    px = 7.6 + np.random.rand() * 1.6
    py = 6.5 + np.random.rand() * 1.2
    c = Circle((px, py), 0.16, facecolor=blue_dark, edgecolor='white', linewidth=1.0)
    ax.add_patch(c)

# Robot removed from dataset as requested

ax.text(8.4, 5.6, 'Training Dataset', fontsize=fs_large, ha='center', va='center', fontweight='bold', color='#333333')

# ============================================================
# LEFT: Thought bubble
# ============================================================
draw_thought_cloud(ax, 0.3, 4.3, 2.4, 1.1)
ax.text(1.5, 4.85, 'Which training data\ninfluence this generation?', fontsize=fs_med, ha='center', va='center', color='#333333')

draw_person(ax, 0.5, 3.6, size=0.4, color='#666666')

# ============================================================
# CENTER: Influence Estimation box
# ============================================================
center_box = FancyBboxPatch((4.0, 4.1), 3.8, 1.1, boxstyle="round,pad=0.02,rounding_size=0.1",
                             facecolor='white', edgecolor='#333333', linewidth=2.5)
ax.add_patch(center_box)
ax.text(5.9, 4.65, 'Influence Estimation', fontsize=fs_xl, ha='center', va='center', fontweight='bold', color='#333333')

# V-shaped arrows pointing into the box
# Left arrow from Test Generation
ax.annotate('', xy=(5.0, 5.2), xytext=(4.4, 5.8),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=2.5))
# Right arrow from Training Dataset
ax.annotate('', xy=(6.8, 5.2), xytext=(7.6, 5.5),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=2.5))

# ============================================================
# Influence scores row
# ============================================================
# Main grouping box or just more space
ax.text(1.8, 3.0, 'Influence:', fontsize=fs_large, ha='center', va='center', fontweight='bold', color='#333333')

# High influence scores (left side)
scores_high = [0.98, 0.83, 0.79]
positions_high = [2.9, 3.8, 4.7]
for x, score in zip(positions_high, scores_high):
    ax.text(x, 3.45, f'{score}', fontsize=fs_med, ha='center', va='center', fontweight='bold', color='#333333')
    c = Circle((x, 3.0), 0.28, facecolor=blue_dark, edgecolor='white', linewidth=1.5)
    ax.add_patch(c)

# Dots
ax.text(5.6, 3.0, '...', fontsize=28, ha='center', va='center', fontweight='bold', color='#333333')

# Low influence scores (right side, faded)
scores_low = [0.13, 0.07, 0.02]
positions_low = [6.5, 7.4, 8.3]
alphas = [0.5, 0.4, 0.25]
for x, score, alpha in zip(positions_low, scores_low, alphas):
    ax.text(x, 3.45, f'{score}', fontsize=fs_med, ha='center', va='center', fontweight='bold', color='#333333', alpha=0.7)
    c = Circle((x, 3.0), 0.28, facecolor=blue_dark, edgecolor='white', linewidth=1.5, alpha=alpha)
    ax.add_patch(c)

# Arrow down from box to scores - lengthened
ax.annotate('', xy=(5.9, 3.8), xytext=(5.9, 4.1),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=2.5))

# ============================================================
# BOTTOM LEFT: High influence example
# ============================================================
left_box = FancyBboxPatch((0.2, 0.3), 4.6, 1.8, boxstyle="round,pad=0.02,rounding_size=0.1",
                           facecolor=gray_light, edgecolor=gray_border, linewidth=2.0)
ax.add_patch(left_box)

# Header with circle
high_circle = Circle((0.6, 1.7), 0.2, facecolor=blue_dark, edgecolor='white', linewidth=1.5)
ax.add_patch(high_circle)
ax.text(0.9, 1.7, 'Influence: 0.98', fontsize=fs_large, ha='left', va='center', fontweight='bold', color=blue_dark)

# Content
ax.text(0.4, 1.3, 'Instruction: Is sublingual varices\nrelated to hypertension?', fontsize=fs_med, ha='left', va='center', color='#333333')
ax.text(0.4, 0.8, 'Output:', fontsize=fs_med, ha='left', va='center', fontweight='bold', color=blue_dark)
ax.text(1.2, 0.8, ' Yes. An association was found\nbetween sublingual varices...', fontsize=fs_med, ha='left', va='center', color=blue_dark)

# Arrow pointing to this box - lengthened and adjusted
ax.annotate('', xy=(2.5, 2.1), xytext=(3.0, 2.7),
            arrowprops=dict(arrowstyle='<-', color=blue_dark, lw=2.0))

# ============================================================
# BOTTOM RIGHT: Low influence example
# ============================================================
right_box = FancyBboxPatch((5.6, 0.3), 4.6, 1.8, boxstyle="round,pad=0.02,rounding_size=0.1",
                            facecolor=gray_light, edgecolor=gray_border, linewidth=2.0)
ax.add_patch(right_box)

# Header with circle (faded)
low_circle = Circle((6.0, 1.7), 0.2, facecolor=blue_dark, edgecolor='white', linewidth=1.5, alpha=0.4)
ax.add_patch(low_circle)
ax.text(6.3, 1.7, 'Influence: 0.07', fontsize=fs_large, ha='left', va='center', fontweight='bold', color=blue_dark)

# Content (PubMedQA example)
ax.text(5.8, 1.3, 'Instruction: Does macular degeneration\nincrease risk of falls?', fontsize=fs_med, ha='left', va='center', color='#333333')
ax.text(5.8, 0.8, 'Output:', fontsize=fs_med, ha='left', va='center', fontweight='bold', color=blue_dark)
ax.text(6.6, 0.8, ' Yes. Patients with advanced AMD\nare at higher risk...', fontsize=fs_med, ha='left', va='center', color=blue_dark)

# Arrow pointing to this box - lengthened and adjusted
ax.annotate('', xy=(7.9, 2.1), xytext=(7.4, 2.7),
            arrowprops=dict(arrowstyle='<-', color=blue_dark, lw=2.0))

plt.tight_layout()
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, '..', 'ICHI_Data_Valuation', 'figures', 'fig0_overview.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print("Overview figure generated successfully!")
