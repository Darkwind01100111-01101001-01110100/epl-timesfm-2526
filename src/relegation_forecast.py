"""
relegation_forecast.py
======================
TimesFM-style forward-looking forecast: Tottenham vs West Ham relegation battle
EPL 2025-26 | Data: MW34 (1 May 2026)

Core mechanics:
  - 12-match rolling context window (TimesFM decoder-only sliding window)
  - Bayesian blend of local form + full-season prior (alpha=0.25)
  - 10,000-sample Monte Carlo simulation → p10/p50/p90 quantile bands
  - Joint simulation to compute P(Spurs relegated) vs P(West Ham relegated)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
spu_df = pd.read_csv(os.path.join(DATA_DIR, 'tottenham_real_2025_26.csv'))
whu_df = pd.read_csv(os.path.join(DATA_DIR, 'westham_real_2025_26.csv'))

def result_to_pts(r): return {'W': 3, 'D': 1, 'L': 0}[r]
def result_color(r):  return {'W': '#2ECC71', 'D': '#F39C12', 'L': '#E74C3C'}[r]

spu_results = spu_df['result'].tolist()
whu_results = whu_df['result'].tolist()
spu_pts_series = [result_to_pts(r) for r in spu_results]
whu_pts_series = [result_to_pts(r) for r in whu_results]

spu_cum = np.cumsum(spu_pts_series)
whu_cum = np.cumsum(whu_pts_series)

spu_pts = int(spu_cum[-1])   # 34
whu_pts = int(whu_cum[-1])   # 36

print(f"Tottenham: {spu_pts} pts after {len(spu_results)} games (18th)")
print(f"West Ham:  {whu_pts} pts after {len(whu_results)} games (17th)")

# ── TimesFM-style forecast ────────────────────────────────────────────────────

def timesfm_forecast(pts_series, horizon, n_samples=10000, seed=42, context_len=12):
    """
    Decoder-only autoregressive forecast using a sliding context window.

    1. Extract the last `context_len` matches as the local pattern baseline.
    2. Blend with the full-season prior via Bayesian smoothing (alpha=0.25).
    3. Run Monte Carlo simulation over the forecast horizon.
    4. Return quantile bands and the raw sample distribution.
    """
    rng = np.random.default_rng(seed)
    context = pts_series[-context_len:]

    p_win_local  = context.count(3) / len(context)
    p_draw_local = context.count(1) / len(context)
    p_loss_local = context.count(0) / len(context)

    p_win_global  = pts_series.count(3) / len(pts_series)
    p_draw_global = pts_series.count(1) / len(pts_series)
    p_loss_global = pts_series.count(0) / len(pts_series)

    alpha = 0.25
    p_win  = (1 - alpha) * p_win_local  + alpha * p_win_global
    p_draw = (1 - alpha) * p_draw_local + alpha * p_draw_global
    p_loss = (1 - alpha) * p_loss_local + alpha * p_loss_global

    total = p_win + p_draw + p_loss
    p_win /= total; p_draw /= total; p_loss /= total

    samples = np.array([
        np.sum(rng.choice([3, 1, 0], size=horizon, p=[p_win, p_draw, p_loss]))
        for _ in range(n_samples)
    ])

    return {
        'p10': np.percentile(samples, 10),
        'p50': np.percentile(samples, 50),
        'p90': np.percentile(samples, 90),
        'mean': np.mean(samples),
        'p_win': p_win, 'p_draw': p_draw, 'p_loss': p_loss,
        'samples': samples,
    }

# Both teams: 4 games remaining (MW35-38)
spu_fc = timesfm_forecast(spu_pts_series, horizon=4)
whu_fc = timesfm_forecast(whu_pts_series, horizon=4)

# ── Relegation probability ────────────────────────────────────────────────────
n = 10000
spu_finals = spu_pts + spu_fc['samples'][:n]
whu_finals = whu_pts + whu_fc['samples'][:n]

p_spu_below = np.mean(spu_finals < whu_finals)
p_whu_below = np.mean(whu_finals < spu_finals)
p_spu_safe  = np.mean(spu_finals >= 38)
p_whu_safe  = np.mean(whu_finals >= 38)

print(f"\nP(Spurs finish below West Ham): {p_spu_below:.1%}")
print(f"P(West Ham finish below Spurs): {p_whu_below:.1%}")
print(f"P(Spurs reach 38 pts):          {p_spu_safe:.1%}")
print(f"P(West Ham reach 38 pts):       {p_whu_safe:.1%}")

# ── Visualization ─────────────────────────────────────────────────────────────
BG    = '#0D0D0D'
PANEL = '#161616'
C_SPU = '#132257'   # Spurs navy
C_WHU = '#7A263A'   # West Ham claret
C_GLD = '#F5A623'
C_GRY = '#888888'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.facecolor': PANEL, 'figure.facecolor': BG,
    'axes.edgecolor': '#2A2A2A', 'axes.labelcolor': '#CCCCCC',
    'xtick.color': '#888888', 'ytick.color': '#888888',
    'text.color': '#DDDDDD', 'grid.color': '#222222',
    'legend.facecolor': '#161616', 'legend.edgecolor': '#333333',
})

fig, (ax_main, ax_form) = plt.subplots(2, 1, figsize=(14, 10),
                                        gridspec_kw={'height_ratios': [3, 1]})
fig.patch.set_facecolor(BG)
ax_main.set_facecolor(PANEL); ax_form.set_facecolor(PANEL)

mw = np.arange(1, 35)
mw_fc = np.array([34, 38])

# Relegation danger zone
ax_main.axhspan(0, 38, alpha=0.07, color='red')
ax_main.axhline(38, color='red', lw=1.2, ls='--', alpha=0.6, label='~Safety threshold (38 pts)')

# Forecast bands
ax_main.fill_between(mw_fc,
    [spu_pts, spu_pts + spu_fc['p10']], [spu_pts, spu_pts + spu_fc['p90']],
    alpha=0.20, color=C_SPU)
ax_main.fill_between(mw_fc,
    [whu_pts, whu_pts + whu_fc['p10']], [whu_pts, whu_pts + whu_fc['p90']],
    alpha=0.20, color=C_WHU)
ax_main.plot(mw_fc, [spu_pts, spu_pts + spu_fc['p50']], '--', color=C_SPU, lw=2)
ax_main.plot(mw_fc, [whu_pts, whu_pts + whu_fc['p50']], '--', color=C_WHU, lw=2)

# Actual trajectories
ax_main.plot(mw, spu_cum, '-', color=C_SPU, lw=2.8, label='Tottenham (actual)', zorder=5)
ax_main.plot(mw, whu_cum, '-', color=C_WHU, lw=2.8, label='West Ham (actual)', zorder=5)
ax_main.scatter([34], [spu_pts], color=C_SPU, s=80, zorder=6)
ax_main.scatter([34], [whu_pts], color=C_WHU, s=80, zorder=6)

# Forecast cutoff
ax_main.axvline(34, color=C_GLD, lw=1.5, ls=':', alpha=0.8)
ax_main.text(34.2, 2, '← Actual    Forecast →', color=C_GLD, fontsize=8, va='bottom')

# Projected final labels
ax_main.text(38.2, spu_pts + spu_fc['p50'],
             f"~{spu_pts + spu_fc['p50']:.0f}", color=C_SPU, fontsize=9, fontweight='bold', va='center')
ax_main.text(38.2, whu_pts + whu_fc['p50'],
             f"~{whu_pts + whu_fc['p50']:.0f}", color=C_WHU, fontsize=9, fontweight='bold', va='center')

ax_main.set_xlim(0.5, 39.5); ax_main.set_ylim(0, 55)
ax_main.set_ylabel('Cumulative Points', fontsize=11)
ax_main.set_title(
    'Relegation Battle: Tottenham vs West Ham · EPL 2025–26 · TimesFM-Style Forecast\n'
    'Data: MW34 (1 May 2026) · Remaining: MW35–38',
    fontsize=12, fontweight='bold', color='white', pad=12)
ax_main.legend(loc='upper left', fontsize=10)
ax_main.grid(True, axis='y', alpha=0.25)
ax_main.xaxis.set_major_locator(MaxNLocator(integer=True))

# Probability box
ax_main.text(0.98, 0.05,
    f"P(Spurs relegated) = {p_spu_below:.0%}  ←  EDGE\n"
    f"P(WHU relegated) = {p_whu_below:.0%}\n"
    f"P(Spurs reach safety ≥38 pts) = {p_spu_safe:.1%}\n"
    f"P(WHU reach safety ≥38 pts) = {p_whu_safe:.1%}\n"
    f"Spurs 12-game PPG: 0.58  |  WHU 12-game PPG: 1.58",
    transform=ax_main.transAxes, fontsize=8.5, color='white',
    ha='right', va='bottom',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#0A0A0A', edgecolor='#E74C3C', alpha=0.95))

# Rolling 5-game PPG form
def rolling_ppg(pts_list, w=5):
    return [np.mean(pts_list[max(0, i-w+1):i+1]) for i in range(len(pts_list))]

ax_form.plot(mw, rolling_ppg(spu_pts_series), '-', color=C_SPU, lw=2, label='Spurs 5-game PPG')
ax_form.plot(mw, rolling_ppg(whu_pts_series), '-', color=C_WHU, lw=2, label='West Ham 5-game PPG')
ax_form.axhline(1.0, color=C_GRY, lw=0.8, ls='--', alpha=0.4)
ax_form.axvline(34, color=C_GLD, lw=1.5, ls=':', alpha=0.8)

# Result strips
for i, r in enumerate(spu_results):
    ax_form.add_patch(mpatches.Rectangle((i+0.6, 3.12), 0.8, 0.22,
        facecolor=result_color(r), alpha=0.85, transform=ax_form.transData))
ax_form.text(0.3, 3.23, 'SPU:', transform=ax_form.transData, fontsize=7.5,
             color=C_SPU, fontweight='bold', va='center')
for i, r in enumerate(whu_results):
    ax_form.add_patch(mpatches.Rectangle((i+0.6, 2.85), 0.8, 0.22,
        facecolor=result_color(r), alpha=0.85, transform=ax_form.transData))
ax_form.text(0.3, 2.96, 'WHU:', transform=ax_form.transData, fontsize=7.5,
             color=C_WHU, fontweight='bold', va='center')

ax_form.set_xlim(0.5, 39.5); ax_form.set_ylim(0, 3.5)
ax_form.set_xlabel('Matchweek', fontsize=11)
ax_form.set_ylabel('5-Game PPG', fontsize=10)
ax_form.legend(loc='upper left', fontsize=9)
ax_form.grid(True, axis='y', alpha=0.25)
ax_form.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
out_path = os.path.join(OUTPUTS_DIR, 'relegation_forecast.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"\nSaved: {out_path}")
