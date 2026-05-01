"""
arsenal_title_forecast.py
=========================
TimesFM-style forward-looking forecast: Arsenal title race vs Manchester City
EPL 2025-26 | Data: MW34 (1 May 2026)

Core mechanics:
  - 12-match rolling context window (TimesFM decoder-only sliding window)
  - Bayesian blend of local form + full-season prior (alpha=0.25)
  - 10,000-sample Monte Carlo simulation → p10/p50/p90 quantile bands
  - Joint simulation to compute P(Arsenal title) vs P(Man City title)
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
ars_df = pd.read_csv(os.path.join(DATA_DIR, 'arsenal_real_2025_26.csv'))
# Man City data is embedded here (no separate CSV in repo yet)
# MW1-33 confirmed results (21W-7D-5L = 70 pts, MW31 postponed to MW31 makeup)
mci_results = [
    'W','L','L','W','D','W','W','W','L','W',
    'W','L','W','W','W','W','W','W','D','D',
    'D','L','W','D','W','W','W','W','D','D',
    'W','W','W'
]

def result_to_pts(r): return {'W': 3, 'D': 1, 'L': 0}[r]
def result_color(r):  return {'W': '#2ECC71', 'D': '#F39C12', 'L': '#E74C3C'}[r]

ars_results = ars_df['result'].tolist()
ars_pts_series = [result_to_pts(r) for r in ars_results]
mci_pts_series = [result_to_pts(r) for r in mci_results]

ars_cum = np.cumsum(ars_pts_series)
mci_cum = np.cumsum(mci_pts_series)

ars_pts = int(ars_cum[-1])   # 73
mci_pts = int(mci_cum[-1])   # 70

print(f"Arsenal:   {ars_pts} pts after {len(ars_results)} games")
print(f"Man City:  {mci_pts} pts after {len(mci_results)} games (1 game in hand)")

# ── TimesFM-style forecast ────────────────────────────────────────────────────

def timesfm_forecast(pts_series, horizon, n_samples=10000, seed=42, context_len=12):
    """
    Decoder-only autoregressive forecast using a sliding context window.

    1. Extract the last `context_len` matches as the local pattern baseline.
    2. Blend with the full-season prior via Bayesian smoothing (alpha=0.25).
    3. Run Monte Carlo simulation over the forecast horizon.
    4. Return p10/p50/p90 quantile bands and the raw sample distribution.
    """
    rng = np.random.default_rng(seed)
    context = pts_series[-context_len:]

    # Local win/draw/loss rates from context window
    p_win_local  = context.count(3) / len(context)
    p_draw_local = context.count(1) / len(context)
    p_loss_local = context.count(0) / len(context)

    # Global prior from full season
    p_win_global  = pts_series.count(3) / len(pts_series)
    p_draw_global = pts_series.count(1) / len(pts_series)
    p_loss_global = pts_series.count(0) / len(pts_series)

    # Bayesian blend (alpha=0.25 → 75% local, 25% global)
    alpha = 0.25
    p_win  = (1 - alpha) * p_win_local  + alpha * p_win_global
    p_draw = (1 - alpha) * p_draw_local + alpha * p_draw_global
    p_loss = (1 - alpha) * p_loss_local + alpha * p_loss_global

    # Normalize
    total = p_win + p_draw + p_loss
    p_win /= total; p_draw /= total; p_loss /= total

    # Monte Carlo
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

# Arsenal: 4 games remaining (MW35-38)
ars_fc = timesfm_forecast(ars_pts_series, horizon=4)
# Man City: 5 games remaining (MW31 makeup + MW35-38)
mci_fc = timesfm_forecast(mci_pts_series, horizon=5)

# ── Title probability ─────────────────────────────────────────────────────────
n = 10000
ars_finals = ars_pts + ars_fc['samples'][:n]
mci_finals = mci_pts + mci_fc['samples'][:n]

p_ars_ahead = np.mean(ars_finals > mci_finals)
p_mci_ahead = np.mean(mci_finals > ars_finals)
p_equal     = np.mean(ars_finals == mci_finals)

# Ties broken by goal difference: Arsenal +38 vs City +37 → Arsenal wins
p_ars_title = p_ars_ahead + p_equal
p_mci_title = p_mci_ahead

print(f"\nP(Arsenal title):   {p_ars_title:.1%}")
print(f"P(Man City title):  {p_mci_title:.1%}")
print(f"(Ties go to Arsenal on GD: +38 vs +37)")

# ── Visualization ─────────────────────────────────────────────────────────────
BG    = '#0D0D0D'
PANEL = '#161616'
C_ARS = '#063672'   # Arsenal navy
C_MCI = '#6CABDD'   # City sky blue
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

mw_ars = np.arange(1, len(ars_results) + 1)
mw_mci = np.arange(1, len(mci_results) + 1)

# Forecast bands
mw_ars_fc = np.array([34, 38])
mw_mci_fc = np.array([33, 38])
ax_main.fill_between(mw_ars_fc,
    [ars_pts, ars_pts + ars_fc['p10']], [ars_pts, ars_pts + ars_fc['p90']],
    alpha=0.18, color=C_ARS)
ax_main.fill_between(mw_mci_fc,
    [mci_pts, mci_pts + mci_fc['p10']], [mci_pts, mci_pts + mci_fc['p90']],
    alpha=0.18, color=C_MCI)
ax_main.plot(mw_ars_fc, [ars_pts, ars_pts + ars_fc['p50']], '--', color=C_ARS, lw=2)
ax_main.plot(mw_mci_fc, [mci_pts, mci_pts + mci_fc['p50']], '--', color=C_MCI, lw=2)

# Actual trajectories
ax_main.plot(mw_ars, ars_cum, '-', color=C_ARS, lw=2.8, label='Arsenal (actual)', zorder=5)
ax_main.plot(mw_mci, mci_cum, '-', color=C_MCI, lw=2.8, label='Man City (actual)', zorder=5)
ax_main.scatter([34], [ars_pts], color=C_ARS, s=80, zorder=6)
ax_main.scatter([33], [mci_pts], color=C_MCI, s=80, zorder=6)

# Forecast cutoff line
ax_main.axvline(34, color=C_GLD, lw=1.5, ls=':', alpha=0.8)
ax_main.text(34.2, 8, '← Actual    Forecast →', color=C_GLD, fontsize=8, va='bottom')

# Projected final labels
ax_main.text(38.2, ars_pts + ars_fc['p50'],
             f"~{ars_pts + ars_fc['p50']:.0f}", color=C_ARS, fontsize=9, fontweight='bold', va='center')
ax_main.text(38.2, mci_pts + mci_fc['p50'],
             f"~{mci_pts + mci_fc['p50']:.0f}", color=C_MCI, fontsize=9, fontweight='bold', va='center')

ax_main.set_xlim(0.5, 39.5); ax_main.set_ylim(0, 95)
ax_main.set_ylabel('Cumulative Points', fontsize=11)
ax_main.set_title(
    'Arsenal Title Race · EPL 2025–26 · TimesFM-Style Forecast\n'
    'Data: MW34 (1 May 2026) · Remaining: MW35–38',
    fontsize=12, fontweight='bold', color='white', pad=12)
ax_main.legend(loc='upper left', fontsize=10)
ax_main.grid(True, axis='y', alpha=0.25)
ax_main.xaxis.set_major_locator(MaxNLocator(integer=True))

# Probability box
ax_main.text(0.98, 0.05,
    f"P(Arsenal title) = {p_ars_title:.1%}  ←  EDGE\n"
    f"P(Man City title) = {p_mci_title:.1%}\n"
    f"GD tiebreak: ARS +38 vs MCI +37\n"
    f"City has 1 game in hand (5 remaining)",
    transform=ax_main.transAxes, fontsize=9, color='white',
    ha='right', va='bottom',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#0A0A0A', edgecolor=C_GLD, alpha=0.95))

# Rolling 5-game PPG form
def rolling_ppg(pts_list, w=5):
    return [np.mean(pts_list[max(0, i-w+1):i+1]) for i in range(len(pts_list))]

ax_form.plot(mw_ars, rolling_ppg(ars_pts_series), '-', color=C_ARS, lw=2, label='Arsenal 5-game PPG')
ax_form.plot(mw_mci, rolling_ppg(mci_pts_series), '-', color=C_MCI, lw=2, label='Man City 5-game PPG')
ax_form.axhline(2.0, color=C_GRY, lw=0.8, ls='--', alpha=0.4)
ax_form.axvline(34, color=C_GLD, lw=1.5, ls=':', alpha=0.8)

# Result strips
for i, r in enumerate(ars_results):
    ax_form.add_patch(mpatches.Rectangle((i+0.6, 3.12), 0.8, 0.22,
        facecolor=result_color(r), alpha=0.85, transform=ax_form.transData))
ax_form.text(0.3, 3.23, 'ARS:', transform=ax_form.transData, fontsize=7.5,
             color=C_ARS, fontweight='bold', va='center')
for i, r in enumerate(mci_results):
    ax_form.add_patch(mpatches.Rectangle((i+0.6, 2.85), 0.8, 0.22,
        facecolor=result_color(r), alpha=0.85, transform=ax_form.transData))
ax_form.text(0.3, 2.96, 'MCI:', transform=ax_form.transData, fontsize=7.5,
             color=C_MCI, fontweight='bold', va='center')

ax_form.set_xlim(0.5, 39.5); ax_form.set_ylim(0, 3.5)
ax_form.set_xlabel('Matchweek', fontsize=11)
ax_form.set_ylabel('5-Game PPG', fontsize=10)
ax_form.legend(loc='lower left', fontsize=9)
ax_form.grid(True, axis='y', alpha=0.25)
ax_form.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
out_path = os.path.join(OUTPUTS_DIR, 'arsenal_title_forecast.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"\nSaved: {out_path}")
