"""
carousel.py
------------
Generates a 4-slide square carousel (1080x1080px each) as individual PNGs
and a combined multi-page PDF for LinkedIn / Threads document upload.

Slide 1: The Hook — The chart + the core tension
Slide 2: The TL;DR — What TimesFM is and why it matters here
Slide 3: The Code — Core implementation snippet
Slide 4: The Signal — What to watch, next steps
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import timesfm
from timesfm.timesfm_2p5 import timesfm_2p5_torch

# ── Shared style ──────────────────────────────────────────────────────────────
CHELSEA_BLUE  = '#034694'
FORECAST_ORG  = '#FFA500'
BG_DARK       = '#0d1117'
BG_MID        = '#161b22'
TEXT_WHITE    = '#f0f6fc'
TEXT_MUTED    = '#8b949e'
ACCENT_GREEN  = '#3fb950'
ACCENT_RED    = '#f85149'
SLIDE_SIZE    = (10.8, 10.8)

# ── Load data + run forecast ──────────────────────────────────────────────────
df = pd.read_csv('../data/chelsea_real_2025_26.csv', parse_dates=['date'])
df = df.sort_values('matchweek').reset_index(drop=True)
points_map = {'W': 3, 'D': 1, 'L': 0}
df['points_earned'] = df['result'].map(points_map)
df['cumulative_points'] = df['points_earned'].cumsum()
current_pts = int(df['cumulative_points'].iloc[-1])

print("Loading TimesFM 2.5...")
model = timesfm_2p5_torch.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)
model.compile(timesfm.ForecastConfig(
    max_context=512, max_horizon=10,
    normalize_inputs=True, use_continuous_quantile_head=True,
    force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True,
))
pt_fc, qt_fc = model.forecast(
    horizon=7, inputs=[df['points_earned'].values.astype(float)]
)
mean_pts  = current_pts + np.cumsum(pt_fc[0])
lower_pts = current_pts + np.cumsum(qt_fc[0, :, 0])
upper_pts = current_pts + np.cumsum(qt_fc[0, :, -1])
future_mw = list(range(32, 39))
proj_mean  = int(round(mean_pts[-1]))
proj_lower = int(round(lower_pts[-1]))
proj_upper = int(round(upper_pts[-1]))
linear_proj = round(current_pts / 31 * 38)
print(f"Forecast: {proj_mean} pts ({proj_lower}–{proj_upper})")

slides = []

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 1: The Hook — Chart + tension
# ─────────────────────────────────────────────────────────────────────────────
fig1, ax = plt.subplots(figsize=SLIDE_SIZE)
fig1.patch.set_facecolor(BG_DARK)
ax.set_facecolor(BG_MID)

ax.axhspan(66, 80, alpha=0.07, color='#1a73e8', zorder=0)
ax.axhspan(60, 66, alpha=0.09, color='#ff6d00', zorder=0)
ax.axhspan(55, 60, alpha=0.09, color='#2ca02c', zorder=0)
ax.axhline(66, color='#1a73e8', linewidth=0.9, linestyle='--', alpha=0.6)
ax.axhline(60, color='#ff6d00', linewidth=0.9, linestyle='--', alpha=0.6)
ax.axhline(55, color='#2ca02c', linewidth=0.9, linestyle='--', alpha=0.6)
ax.text(39.2, 66.4, 'CL',   color='#5b9cf6', fontsize=9, va='bottom', fontweight='bold')
ax.text(39.2, 60.4, 'EL',   color='#ff9d45', fontsize=9, va='bottom', fontweight='bold')
ax.text(39.2, 55.4, 'UECL', color='#56d364', fontsize=9, va='bottom', fontweight='bold')

ax.plot(df['matchweek'], df['cumulative_points'],
        color=CHELSEA_BLUE, linewidth=2.8, marker='o', markersize=3.5,
        label='Actual (MW1–31)', zorder=4)
ax.plot([31, 32], [current_pts, mean_pts[0]],
        color=FORECAST_ORG, linewidth=1.5, linestyle='--', zorder=3)
ax.plot(future_mw, mean_pts,
        color=FORECAST_ORG, linewidth=2.8, linestyle='--', marker='o', markersize=5,
        label=f'TimesFM: ~{proj_mean} pts', zorder=4)
ax.fill_between(future_mw, lower_pts, upper_pts,
                color=FORECAST_ORG, alpha=0.2, label=f'80% CI ({proj_lower}–{proj_upper})')
ax.scatter([38], [linear_proj], color=TEXT_MUTED, marker='x', s=80, zorder=5,
           label=f'Linear: {linear_proj} pts')
ax.annotate(f'Now: {current_pts} pts · 6th',
            xy=(31, current_pts), xytext=(22, current_pts + 8),
            arrowprops=dict(arrowstyle='->', color=CHELSEA_BLUE, lw=1.3),
            fontsize=10, color='#6fa8dc', fontweight='bold')
ax.annotate(f'TimesFM: ~{proj_mean}',
            xy=(38, mean_pts[-1]), xytext=(33.5, mean_pts[-1] + 5),
            arrowprops=dict(arrowstyle='->', color=FORECAST_ORG, lw=1.3),
            fontsize=10, color=FORECAST_ORG)

ax.set_xlim(0, 41)
ax.set_ylim(0, 78)
ax.set_xticks(range(0, 40, 4))
ax.tick_params(colors=TEXT_MUTED, labelsize=9)
ax.spines[['top','right','bottom','left']].set_color('#30363d')
ax.set_xlabel('Matchweek', fontsize=10, color=TEXT_MUTED)
ax.set_ylabel('Cumulative Points', fontsize=10, color=TEXT_MUTED)
ax.grid(True, color='#21262d', linewidth=0.6)
ax.legend(loc='upper left', fontsize=9, framealpha=0.3,
          facecolor=BG_MID, edgecolor='#30363d', labelcolor=TEXT_WHITE)

fig1.text(0.5, 0.97, 'Chelsea FC 2025–26', ha='center', va='top',
          fontsize=17, color=TEXT_WHITE, fontweight='bold')
fig1.text(0.5, 0.93, 'Simple math says 59 pts.  The model says 56.',
          ha='center', va='top', fontsize=12, color=TEXT_MUTED)
fig1.text(0.5, 0.02, 'Data: FBref · MW1–31 · Apr 2026  |  Model: TimesFM 2.5 (Google)',
          ha='center', va='bottom', fontsize=8, color=TEXT_MUTED)

plt.subplots_adjust(top=0.88, bottom=0.10, left=0.09, right=0.88)
plt.savefig('../outputs/carousel_slide1.png', dpi=100, bbox_inches='tight', facecolor=BG_DARK)
slides.append(fig1)
print("Slide 1 done")

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 2: What is TimesFM — fixed layout, no overlaps
# ─────────────────────────────────────────────────────────────────────────────
fig2 = plt.figure(figsize=SLIDE_SIZE)
fig2.patch.set_facecolor(BG_DARK)
ax2 = fig2.add_axes([0, 0, 1, 1])
ax2.set_facecolor(BG_DARK)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')

# Title block
ax2.text(0.5, 0.94, 'What is TimesFM?', ha='center', va='top',
         fontsize=24, color=TEXT_WHITE, fontweight='bold')
ax2.text(0.5, 0.87, "Google's zero-shot time-series foundation model",
         ha='center', va='top', fontsize=13, color=FORECAST_ORG)

# Divider line
ax2.plot([0.08, 0.92], [0.82, 0.82], color='#30363d', linewidth=1.2)

# Three key points — fixed y positions with generous spacing
items = [
    (0.74, '200M parameters',
     'Pre-trained on 100 billion real-world time points.\nNo fine-tuning required on your specific data.'),
    (0.56, 'Zero-shot forecasting',
     'Feed it any sequence of numbers — match points,\nstock prices, temperature — and it forecasts forward.'),
    (0.38, 'Uncertainty-aware',
     'Returns a full distribution, not a single number.\nThe 80% confidence interval is the honest read.'),
]
for y_title, title, body in items:
    ax2.text(0.09, y_title, '▸', ha='left', va='top',
             fontsize=16, color=FORECAST_ORG)
    ax2.text(0.16, y_title, title, ha='left', va='top',
             fontsize=14, color=TEXT_WHITE, fontweight='bold')
    ax2.text(0.16, y_title - 0.065, body, ha='left', va='top',
             fontsize=11, color=TEXT_MUTED, linespacing=1.55)

# Bottom callout box — positioned with fixed clearance from text above
callout_y = 0.10
callout_h = 0.14
box = FancyBboxPatch((0.08, callout_y), 0.84, callout_h,
                     boxstyle="round,pad=0.02",
                     facecolor='#0d2818', edgecolor='#2ea043', linewidth=1.5,
                     zorder=2)
ax2.add_patch(box)
ax2.text(0.5, callout_y + callout_h - 0.02,
         'Why use it for football?',
         ha='center', va='top', fontsize=12, color=ACCENT_GREEN,
         fontweight='bold', zorder=3)
ax2.text(0.5, callout_y + callout_h - 0.065,
         'A 38-game season is a short, noisy sequence.\nTimesFM was built for exactly this kind of data.',
         ha='center', va='top', fontsize=11, color=TEXT_WHITE,
         linespacing=1.5, zorder=3)

# Footer
ax2.text(0.5, 0.015, 'github.com/Darkwind01100111-01101001-01110100/chelsea-timesfm',
         ha='center', va='bottom', fontsize=8, color=TEXT_MUTED)

plt.savefig('../outputs/carousel_slide2.png', dpi=100, bbox_inches='tight', facecolor=BG_DARK)
slides.append(fig2)
print("Slide 2 done")

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 3: The Code — fixed, text visible inside box
# ─────────────────────────────────────────────────────────────────────────────
fig3 = plt.figure(figsize=SLIDE_SIZE)
fig3.patch.set_facecolor(BG_DARK)
ax3 = fig3.add_axes([0, 0, 1, 1])
ax3.set_facecolor(BG_DARK)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis('off')

# Title
ax3.text(0.5, 0.95, 'The Core Implementation', ha='center', va='top',
         fontsize=21, color=TEXT_WHITE, fontweight='bold')
ax3.text(0.5, 0.88,
         'Forecast per-match points (0/1/3),\nthen accumulate from the current baseline.',
         ha='center', va='top', fontsize=11, color=TEXT_MUTED, linespacing=1.5)

# Code block background
code_bg = FancyBboxPatch((0.05, 0.22), 0.90, 0.60,
                          boxstyle="round,pad=0.02",
                          facecolor='#161b22', edgecolor='#30363d', linewidth=1.2,
                          zorder=1)
ax3.add_patch(code_bg)

# Code text — rendered inside the box
code = (
    "# Forecast per-match points (0, 1, or 3)\n"
    "pt_fc, qt_fc = model.forecast(\n"
    "    horizon=7,\n"
    "    inputs=[df['points_earned'].values]\n"
    ")\n\n"
    "# Accumulate on top of current 48 pts\n"
    "per_match = pt_fc[0]\n"
    "projected  = current_pts + np.cumsum(per_match)"
)
ax3.text(0.10, 0.79, code,
         ha='left', va='top', fontsize=12,
         color=TEXT_WHITE, fontfamily='monospace',
         linespacing=1.75, zorder=2)

# Output line — highlighted box inside the code block
out_bg = FancyBboxPatch((0.06, 0.23), 0.88, 0.075,
                         boxstyle="round,pad=0.01",
                         facecolor='#0d2818', edgecolor='#2ea043', linewidth=1.2,
                         zorder=2)
ax3.add_patch(out_bg)
ax3.text(0.10, 0.268,
         f'>>> projected[-1]   →   {proj_mean} pts',
         ha='left', va='center', fontsize=13,
         color=ACCENT_GREEN, fontfamily='monospace',
         fontweight='bold', zorder=3)

# Insight note below the box
ax3.text(0.5, 0.18,
         'Why not forecast the cumulative line directly?',
         ha='center', va='top', fontsize=11, color=TEXT_WHITE,
         fontweight='bold')
ax3.text(0.5, 0.13,
         'The last 3 matches were losses → flat tail → model predicts "level" not growth.',
         ha='center', va='top', fontsize=10, color=TEXT_MUTED,
         style='italic')

# Footer
ax3.text(0.5, 0.015, 'github.com/Darkwind01100111-01101001-01110100/chelsea-timesfm',
         ha='center', va='bottom', fontsize=8, color=TEXT_MUTED)

plt.savefig('../outputs/carousel_slide3.png', dpi=100, bbox_inches='tight', facecolor=BG_DARK)
slides.append(fig3)
print("Slide 3 done")

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 4: What to Watch — fixed alignment, no overlaps
# ─────────────────────────────────────────────────────────────────────────────
fig4 = plt.figure(figsize=SLIDE_SIZE)
fig4.patch.set_facecolor(BG_DARK)
ax4 = fig4.add_axes([0, 0, 1, 1])
ax4.set_facecolor(BG_DARK)
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')

# Title
ax4.text(0.5, 0.95, 'What to Watch', ha='center', va='top',
         fontsize=24, color=TEXT_WHITE, fontweight='bold')
ax4.text(0.5, 0.88, '7 matches left. Two very different projections.',
         ha='center', va='top', fontsize=13, color=TEXT_MUTED)

# Tension summary box
tension_bg = FancyBboxPatch((0.06, 0.76), 0.88, 0.09,
                             boxstyle="round,pad=0.02",
                             facecolor='#161b22', edgecolor='#30363d', linewidth=1.2,
                             zorder=1)
ax4.add_patch(tension_bg)
ax4.text(0.5, 0.805,
         f'Linear: {linear_proj} pts   ·   TimesFM: {proj_mean} pts   ·   Gap: {linear_proj - proj_mean} pts',
         ha='center', va='center', fontsize=13, color=TEXT_WHITE,
         fontweight='bold', zorder=2)

# Section label
ax4.text(0.08, 0.72, 'Remaining schedule:', ha='left', va='top',
         fontsize=11, color=TEXT_MUTED)

# Fixtures — evenly spaced, no overlap
fixtures = [
    ('MW32', 'Man City (H)',     'hard'),
    ('MW33', 'Man Utd (H)',      'hard'),
    ('MW34', 'Brighton (A)',     'medium'),
    ('MW35', 'Nott Forest (H)', 'soft'),
    ('MW36', 'Liverpool (A)',    'medium'),
    ('MW37', 'Spurs (H)',        'medium'),
    ('MW38', 'Sunderland (A)',   'medium'),
]
diff_color = {'hard': ACCENT_RED, 'medium': FORECAST_ORG, 'soft': ACCENT_GREEN}
diff_label = {'hard': 'Hard', 'medium': 'Medium', 'soft': 'Soft'}

# 7 rows, starting at y=0.67, step=0.074 → bottom row at ~0.67 - 6*0.074 = 0.226
row_start = 0.67
row_step  = 0.074
for i, (mw, opp, diff) in enumerate(fixtures):
    y = row_start - i * row_step
    color = diff_color[diff]
    ax4.text(0.08, y, mw, ha='left', va='center', fontsize=10, color=TEXT_MUTED)
    ax4.text(0.20, y, opp, ha='left', va='center', fontsize=12,
             color=TEXT_WHITE, fontweight='bold')
    # Tag badge
    badge = FancyBboxPatch((0.70, y - 0.022), 0.20, 0.044,
                            boxstyle="round,pad=0.01",
                            facecolor=BG_MID, edgecolor=color, linewidth=1.2,
                            zorder=1)
    ax4.add_patch(badge)
    ax4.text(0.80, y, diff_label[diff], ha='center', va='center',
             fontsize=10, color=color, fontweight='bold', zorder=2)

# Divider above footer
ax4.plot([0.08, 0.92], [0.115, 0.115], color='#30363d', linewidth=0.8)

# Footer
ax4.text(0.5, 0.09,
         'Tracking week over week through May.',
         ha='center', va='top', fontsize=11, color=TEXT_MUTED)
ax4.text(0.5, 0.045,
         'github.com/Darkwind01100111-01101001-01110100/chelsea-timesfm',
         ha='center', va='top', fontsize=9, color=TEXT_MUTED)

plt.savefig('../outputs/carousel_slide4.png', dpi=100, bbox_inches='tight', facecolor=BG_DARK)
slides.append(fig4)
print("Slide 4 done")

# ─────────────────────────────────────────────────────────────────────────────
# Combine into multi-page PDF
# ─────────────────────────────────────────────────────────────────────────────
with PdfPages('../outputs/chelsea_timesfm_carousel.pdf') as pdf:
    for fig in slides:
        pdf.savefig(fig, facecolor=BG_DARK, bbox_inches='tight')
        plt.close(fig)

print("\nAll done:")
for i in range(1, 5):
    print(f"  outputs/carousel_slide{i}.png")
print("  outputs/chelsea_timesfm_carousel.pdf")
