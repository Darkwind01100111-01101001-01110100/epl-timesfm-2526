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
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import timesfm
from timesfm.timesfm_2p5 import timesfm_2p5_torch

# ── Shared style ──────────────────────────────────────────────────────────────
CHELSEA_BLUE  = '#034694'
CHELSEA_GOLD  = '#DBA111'
FORECAST_ORG  = '#FFA500'
BG_DARK       = '#0d1117'
BG_MID        = '#161b22'
TEXT_WHITE    = '#f0f6fc'
TEXT_MUTED    = '#8b949e'
ACCENT_GREEN  = '#3fb950'
ACCENT_RED    = '#f85149'
SLIDE_SIZE    = (10.8, 10.8)   # 1080px at 100dpi

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

# Threshold bands
ax.axhspan(66, 80, alpha=0.07, color='#1a73e8', zorder=0)
ax.axhspan(60, 66, alpha=0.09, color='#ff6d00', zorder=0)
ax.axhspan(55, 60, alpha=0.09, color='#2ca02c', zorder=0)
ax.axhline(66, color='#1a73e8', linewidth=0.9, linestyle='--', alpha=0.6)
ax.axhline(60, color='#ff6d00', linewidth=0.9, linestyle='--', alpha=0.6)
ax.axhline(55, color='#2ca02c', linewidth=0.9, linestyle='--', alpha=0.6)

# Threshold labels
ax.text(39.2, 66.4, 'CL', color='#5b9cf6', fontsize=9, va='bottom', fontweight='bold')
ax.text(39.2, 60.4, 'EL', color='#ff9d45', fontsize=9, va='bottom', fontweight='bold')
ax.text(39.2, 55.4, 'UECL', color='#56d364', fontsize=9, va='bottom', fontweight='bold')

# Actual line
ax.plot(df['matchweek'], df['cumulative_points'],
        color=CHELSEA_BLUE, linewidth=2.8, marker='o', markersize=3.5,
        label='Actual (MW1–31)', zorder=4)

# Connector
ax.plot([31, 32], [current_pts, mean_pts[0]],
        color=FORECAST_ORG, linewidth=1.5, linestyle='--', zorder=3)

# Forecast
ax.plot(future_mw, mean_pts,
        color=FORECAST_ORG, linewidth=2.8, linestyle='--', marker='o', markersize=5,
        label=f'TimesFM: ~{proj_mean} pts', zorder=4)
ax.fill_between(future_mw, lower_pts, upper_pts,
                color=FORECAST_ORG, alpha=0.2, label=f'80% CI ({proj_lower}–{proj_upper})')

# Linear projection
ax.scatter([38], [linear_proj], color=TEXT_MUTED, marker='x', s=80, zorder=5,
           label=f'Linear: {linear_proj} pts')

# Annotations
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
ax.yaxis.label.set_color(TEXT_MUTED)
ax.xaxis.label.set_color(TEXT_MUTED)
ax.set_xlabel('Matchweek', fontsize=10, color=TEXT_MUTED)
ax.set_ylabel('Cumulative Points', fontsize=10, color=TEXT_MUTED)
ax.grid(True, color='#21262d', linewidth=0.6)

legend = ax.legend(loc='upper left', fontsize=9, framealpha=0.3,
                   facecolor=BG_MID, edgecolor='#30363d', labelcolor=TEXT_WHITE)

# Header text above chart
fig1.text(0.5, 0.97, 'Chelsea FC 2025–26', ha='center', va='top',
          fontsize=17, color=TEXT_WHITE, fontweight='bold')
fig1.text(0.5, 0.93, 'Simple math says 59 pts.  The model says 56.',
          ha='center', va='top', fontsize=12, color=TEXT_MUTED)
fig1.text(0.5, 0.02, 'Data: FBref · MW1–31 · Apr 2026  |  Model: TimesFM 2.5 (Google)',
          ha='center', va='bottom', fontsize=8, color=TEXT_MUTED)

plt.subplots_adjust(top=0.88, bottom=0.10, left=0.09, right=0.88)
plt.savefig('../outputs/carousel_slide1.png', dpi=100, bbox_inches='tight',
            facecolor=BG_DARK)
slides.append(fig1)
print("Slide 1 done")

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 2: The TL;DR — What TimesFM is
# ─────────────────────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=SLIDE_SIZE)
fig2.patch.set_facecolor(BG_DARK)
ax2.set_facecolor(BG_DARK)
ax2.axis('off')

# Title
ax2.text(0.5, 0.93, 'What is TimesFM?', ha='center', va='top',
         fontsize=22, color=TEXT_WHITE, fontweight='bold',
         transform=ax2.transAxes)
ax2.text(0.5, 0.86, "Google's zero-shot time-series foundation model",
         ha='center', va='top', fontsize=13, color=FORECAST_ORG,
         transform=ax2.transAxes)

# Divider
ax2.plot([0.1, 0.9], [0.82, 0.82], color='#30363d', linewidth=1,
         transform=ax2.transAxes)

# Three key points
points = [
    ('200M parameters', 'Pre-trained on 100 billion real-world time points.\nNo fine-tuning required on your specific data.'),
    ('Zero-shot forecasting', 'Feed it any sequence of numbers — match points,\nstock prices, temperature — and it forecasts forward.'),
    ('Uncertainty-aware', 'Returns a full distribution, not a single number.\nThe 80% confidence interval is the honest read.'),
]
y_positions = [0.70, 0.52, 0.34]
for (title, body), y in zip(points, y_positions):
    # Accent dot
    ax2.text(0.10, y + 0.025, '▸', ha='left', va='center',
             fontsize=16, color=FORECAST_ORG, transform=ax2.transAxes)
    ax2.text(0.16, y + 0.025, title, ha='left', va='center',
             fontsize=14, color=TEXT_WHITE, fontweight='bold',
             transform=ax2.transAxes)
    ax2.text(0.16, y - 0.015, body, ha='left', va='top',
             fontsize=11, color=TEXT_MUTED, transform=ax2.transAxes,
             linespacing=1.5)

# Bottom note
ax2.text(0.5, 0.10,
         'Why use it for football?\nA 38-game season is a short, noisy sequence.\nTimesFM was built for exactly this kind of data.',
         ha='center', va='top', fontsize=12, color='#56d364',
         transform=ax2.transAxes, linespacing=1.6,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#0d2818', edgecolor='#2ea043', alpha=0.8))

ax2.text(0.5, 0.02, 'github.com/Darkwind01100111-01101001-01110100/chelsea-timesfm',
         ha='center', va='bottom', fontsize=8, color=TEXT_MUTED,
         transform=ax2.transAxes)

plt.savefig('../outputs/carousel_slide2.png', dpi=100, bbox_inches='tight',
            facecolor=BG_DARK)
slides.append(fig2)
print("Slide 2 done")

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 3: The Code — Core implementation
# ─────────────────────────────────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=SLIDE_SIZE)
fig3.patch.set_facecolor(BG_DARK)
ax3.set_facecolor(BG_DARK)
ax3.axis('off')

ax3.text(0.5, 0.95, 'The Core Implementation', ha='center', va='top',
         fontsize=20, color=TEXT_WHITE, fontweight='bold',
         transform=ax3.transAxes)
ax3.text(0.5, 0.89,
         'The key insight: forecast per-match points (0/1/3),\nthen accumulate from the current baseline.',
         ha='center', va='top', fontsize=11, color=TEXT_MUTED,
         transform=ax3.transAxes, linespacing=1.5)

# Code block background
code_box = FancyBboxPatch((0.05, 0.12), 0.90, 0.70,
                           boxstyle="round,pad=0.02",
                           facecolor='#161b22', edgecolor='#30363d',
                           transform=ax3.transAxes, zorder=1)
ax3.add_patch(code_box)

# Code text — using monospace, syntax-colored manually
code_lines = [
    ('# Forecast per-match points (0, 1, or 3)',          TEXT_MUTED,   False),
    ('pt_fc, qt_fc = model.forecast(',                     TEXT_WHITE,   False),
    ('    horizon=7,',                                     '#79c0ff',    False),
    ('    inputs=[df[',                                    TEXT_WHITE,   False),
    ("    inputs=[df['points_earned'].values]",            TEXT_WHITE,   False),
    (')',                                                  TEXT_WHITE,   False),
    ('',                                                   TEXT_WHITE,   False),
    ('# Accumulate on top of current 48 pts',             TEXT_MUTED,   False),
    ('per_match_mean = pt_fc[0]',                         TEXT_WHITE,   False),
    ('projected = current_pts + np.cumsum(per_match_mean)', TEXT_WHITE, False),
    ('',                                                   TEXT_WHITE,   False),
    ('# Output',                                          TEXT_MUTED,   False),
    (f'>>> projected[-1]  →  {proj_mean} pts',            ACCENT_GREEN, True),
]

# Simplified clean code block
code_text = (
    "# Forecast per-match points (0, 1, or 3)\n"
    "pt_fc, qt_fc = model.forecast(\n"
    "    horizon=7,\n"
    "    inputs=[df['points_earned'].values]\n"
    ")\n\n"
    "# Accumulate on top of current 48 pts\n"
    "per_match = pt_fc[0]\n"
    "projected = current_pts + np.cumsum(per_match)\n\n"
    f">>> projected[-1]  →  {proj_mean} pts"
)

ax3.text(0.10, 0.76, code_text,
         ha='left', va='top', fontsize=11.5,
         color=TEXT_WHITE, fontfamily='monospace',
         transform=ax3.transAxes, linespacing=1.7, zorder=2)

# Highlight the output line
highlight = FancyBboxPatch((0.06, 0.13), 0.88, 0.065,
                            boxstyle="round,pad=0.01",
                            facecolor='#0d2818', edgecolor='#2ea043', alpha=0.6,
                            transform=ax3.transAxes, zorder=1)
ax3.add_patch(highlight)

ax3.text(0.5, 0.06,
         'Why not forecast the cumulative line directly?\n'
         'The last 3 matches were losses → flat tail → model predicts "level" not growth.',
         ha='center', va='top', fontsize=10, color=TEXT_MUTED,
         transform=ax3.transAxes, linespacing=1.5,
         style='italic')

plt.savefig('../outputs/carousel_slide3.png', dpi=100, bbox_inches='tight',
            facecolor=BG_DARK)
slides.append(fig3)
print("Slide 3 done")

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 4: The Signal — What to watch
# ─────────────────────────────────────────────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=SLIDE_SIZE)
fig4.patch.set_facecolor(BG_DARK)
ax4.set_facecolor(BG_DARK)
ax4.axis('off')

ax4.text(0.5, 0.95, 'What to Watch', ha='center', va='top',
         fontsize=22, color=TEXT_WHITE, fontweight='bold',
         transform=ax4.transAxes)
ax4.text(0.5, 0.88, '7 matches left. Two very different projections.',
         ha='center', va='top', fontsize=13, color=TEXT_MUTED,
         transform=ax4.transAxes)

# The tension box
tension_box = FancyBboxPatch((0.08, 0.72), 0.84, 0.12,
                              boxstyle="round,pad=0.02",
                              facecolor='#161b22', edgecolor='#30363d',
                              transform=ax4.transAxes)
ax4.add_patch(tension_box)
ax4.text(0.50, 0.80,
         f'Linear projection: {linear_proj} pts   ·   TimesFM: {proj_mean} pts   ·   Gap: {linear_proj - proj_mean} pts',
         ha='center', va='center', fontsize=12, color=TEXT_WHITE,
         fontweight='bold', transform=ax4.transAxes)

# Remaining fixtures
fixtures = [
    ('MW32', 'Man City (H)', 'hard'),
    ('MW33', 'Man Utd (H)',  'hard'),
    ('MW34', 'Brighton (A)', 'medium'),
    ('MW35', 'Nott Forest (H)', 'soft'),
    ('MW36', 'Liverpool (A)', 'medium'),
    ('MW37', 'Spurs (H)',    'medium'),
    ('MW38', 'Sunderland (A)', 'medium'),
]
diff_color = {'hard': ACCENT_RED, 'medium': FORECAST_ORG, 'soft': ACCENT_GREEN}
diff_label = {'hard': 'Hard', 'medium': 'Medium', 'soft': 'Soft'}

ax4.text(0.10, 0.68, 'Remaining schedule:', ha='left', va='top',
         fontsize=11, color=TEXT_MUTED, transform=ax4.transAxes)

for i, (mw, opp, diff) in enumerate(fixtures):
    y = 0.62 - i * 0.072
    color = diff_color[diff]
    ax4.text(0.10, y, mw, ha='left', va='center', fontsize=10,
             color=TEXT_MUTED, transform=ax4.transAxes)
    ax4.text(0.22, y, opp, ha='left', va='center', fontsize=11,
             color=TEXT_WHITE, fontweight='bold', transform=ax4.transAxes)
    ax4.text(0.72, y, diff_label[diff], ha='left', va='center', fontsize=10,
             color=color, fontweight='bold', transform=ax4.transAxes,
             bbox=dict(boxstyle='round,pad=0.25', facecolor=BG_MID,
                       edgecolor=color, alpha=0.7))

# Bottom CTA
ax4.text(0.5, 0.06,
         'Tracking this week over week through May.\nFull code + data: github.com/Darkwind01100111-01101001-01110100/chelsea-timesfm',
         ha='center', va='top', fontsize=10, color=TEXT_MUTED,
         transform=ax4.transAxes, linespacing=1.6)

ax4.plot([0.1, 0.9], [0.10, 0.10], color='#30363d', linewidth=0.8,
         transform=ax4.transAxes)

plt.savefig('../outputs/carousel_slide4.png', dpi=100, bbox_inches='tight',
            facecolor=BG_DARK)
slides.append(fig4)
print("Slide 4 done")

# ─────────────────────────────────────────────────────────────────────────────
# Combine into multi-page PDF
# ─────────────────────────────────────────────────────────────────────────────
with PdfPages('../outputs/chelsea_timesfm_carousel.pdf') as pdf:
    for fig in slides:
        pdf.savefig(fig, facecolor=BG_DARK, bbox_inches='tight')
        plt.close(fig)

print("\nAll done. Files saved:")
print("  outputs/carousel_slide1.png")
print("  outputs/carousel_slide2.png")
print("  outputs/carousel_slide3.png")
print("  outputs/carousel_slide4.png")
print("  outputs/chelsea_timesfm_carousel.pdf")
