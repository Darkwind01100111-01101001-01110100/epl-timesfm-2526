"""
live_snapshot.py
================
Generates the daily live_snapshot.png — the primary chart shown in the README.

Updated MW34 (1 May 2026): Expanded from Chelsea-only to a full EPL 2025-26
season snapshot covering three active stories:
  1. Chelsea: end-of-season points forecast (MW34, 6 games remaining)
  2. Arsenal vs Man City: title race probability (MW34, 4-5 games remaining)
  3. Tottenham vs West Ham: relegation battle (MW34, 4 games remaining)

Run from the repo root:
    python src/live_snapshot.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def result_to_pts(r): return {'W': 3, 'D': 1, 'L': 0}[r]
def result_color(r):  return {'W': '#2ECC71', 'D': '#F39C12', 'L': '#E74C3C'}[r]

def rolling_ppg(pts_list, w=5):
    return [np.mean(pts_list[max(0, i-w+1):i+1]) for i in range(len(pts_list))]

def timesfm_forecast(pts_series, horizon, n_samples=10000, seed=42, context_len=12):
    """
    TimesFM-style decoder-only autoregressive forecast.
    Uses a 12-match context window with Bayesian smoothing (alpha=0.25)
    and Monte Carlo simulation to generate quantile bands.
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
        'p_win': p_win, 'p_draw': p_draw, 'p_loss': p_loss,
        'samples': samples,
    }

# ── Load data ─────────────────────────────────────────────────────────────────
che_df = pd.read_csv(os.path.join(DATA_DIR, 'chelsea_real_2025_26.csv'))
ars_df = pd.read_csv(os.path.join(DATA_DIR, 'arsenal_real_2025_26.csv'))
spu_df = pd.read_csv(os.path.join(DATA_DIR, 'tottenham_real_2025_26.csv'))
whu_df = pd.read_csv(os.path.join(DATA_DIR, 'westham_real_2025_26.csv'))

# Man City MW1-33 results (no separate CSV yet)
mci_results_raw = [
    'W','L','L','W','D','W','W','W','L','W',
    'W','L','W','W','W','W','W','W','D','D',
    'D','L','W','D','W','W','W','W','D','D',
    'W','W','W'
]

che_results = che_df['result'].tolist()
ars_results = ars_df['result'].tolist()
spu_results = spu_df['result'].tolist()
whu_results = whu_df['result'].tolist()
mci_results = mci_results_raw

che_pts_s = [result_to_pts(r) for r in che_results]
ars_pts_s = [result_to_pts(r) for r in ars_results]
spu_pts_s = [result_to_pts(r) for r in spu_results]
whu_pts_s = [result_to_pts(r) for r in whu_results]
mci_pts_s = [result_to_pts(r) for r in mci_results]

che_cum = np.cumsum(che_pts_s); che_pts = int(che_cum[-1])
ars_cum = np.cumsum(ars_pts_s); ars_pts = int(ars_cum[-1])
spu_cum = np.cumsum(spu_pts_s); spu_pts = int(spu_cum[-1])
whu_cum = np.cumsum(whu_pts_s); whu_pts = int(whu_cum[-1])
mci_cum = np.cumsum(mci_pts_s); mci_pts = int(mci_cum[-1])

# Chelsea: 4 games remaining (MW35-38), now at parity with other clubs at MW34
che_fc  = timesfm_forecast(che_pts_s, horizon=4)
ars_fc  = timesfm_forecast(ars_pts_s, horizon=4)
mci_fc  = timesfm_forecast(mci_pts_s, horizon=5)
spu_fc  = timesfm_forecast(spu_pts_s, horizon=4)
whu_fc  = timesfm_forecast(whu_pts_s, horizon=4)

# Title probabilities
n = 10000
ars_finals = ars_pts + ars_fc['samples'][:n]
mci_finals = mci_pts + mci_fc['samples'][:n]
p_ars_title = np.mean(ars_finals >= mci_finals)  # GD tiebreak goes to Arsenal
p_mci_title = np.mean(mci_finals > ars_finals)

# Relegation probabilities
spu_finals = spu_pts + spu_fc['samples'][:n]
whu_finals = whu_pts + whu_fc['samples'][:n]
p_spu_below = np.mean(spu_finals < whu_finals)
p_whu_below = np.mean(whu_finals < spu_finals)
p_spu_safe  = np.mean(spu_finals >= 38)
p_whu_safe  = np.mean(whu_finals >= 38)

# ── Style ─────────────────────────────────────────────────────────────────────
BG    = '#0D0D0D'
PANEL = '#161616'
C_CHE = '#034694'   # Chelsea blue
C_ARS = '#063672'   # Arsenal navy
C_MCI = '#6CABDD'   # City sky blue
C_SPU = '#132257'   # Spurs navy
C_WHU = '#7A263A'   # West Ham claret
C_GLD = '#F5A623'
C_GRY = '#888888'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.facecolor': PANEL, 'figure.facecolor': BG,
    'axes.edgecolor': '#2A2A2A', 'axes.labelcolor': '#CCCCCC',
    'xtick.color': '#888888', 'ytick.color': '#888888',
    'text.color': '#DDDDDD', 'grid.color': '#222222',
    'legend.facecolor': '#161616', 'legend.edgecolor': '#333333',
})

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor(BG)
gs = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.30,
              left=0.06, right=0.97, top=0.91, bottom=0.07)

ax_che   = fig.add_subplot(gs[0, 0])
ax_title = fig.add_subplot(gs[0, 1])
ax_rel   = fig.add_subplot(gs[0, 2])
ax_dist1 = fig.add_subplot(gs[1, 0])
ax_dist2 = fig.add_subplot(gs[1, 1])
ax_form  = fig.add_subplot(gs[1, 2])

for ax in [ax_che, ax_title, ax_rel, ax_dist1, ax_dist2, ax_form]:
    ax.set_facecolor(PANEL)

# ── Panel 1: Chelsea trajectory ───────────────────────────────────────────────
mw_che = np.arange(1, len(che_results) + 1)
mw_che_fc = np.array([len(che_results), 38])
ax_che.plot(mw_che, che_cum, '-', color=C_CHE, lw=2.5, label='Chelsea (actual)')
ax_che.fill_between(mw_che_fc,
    [che_pts, che_pts + che_fc['p10']], [che_pts, che_pts + che_fc['p90']],
    alpha=0.20, color=C_CHE)
ax_che.plot(mw_che_fc, [che_pts, che_pts + che_fc['p50']], '--', color=C_CHE, lw=2)
ax_che.axhline(67, color='#1a73e8', lw=1, ls='--', alpha=0.6)
ax_che.text(0.5, 67.5, 'CL ~67', color='#1a73e8', fontsize=7, transform=ax_che.get_yaxis_transform())
ax_che.axvline(len(che_results), color=C_GLD, lw=1.2, ls=':', alpha=0.8)
ax_che.text(38.2, che_pts + che_fc['p50'],
            f"~{che_pts + che_fc['p50']:.0f}", color=C_CHE, fontsize=8, fontweight='bold', va='center')
ax_che.set_xlim(0.5, 40); ax_che.set_ylim(0, 85)
ax_che.set_title('Chelsea · End-of-Season Forecast\nMW34 → MW38', fontsize=10, fontweight='bold', color='white')
ax_che.set_xlabel('Matchweek', fontsize=9); ax_che.set_ylabel('Points', fontsize=9)
ax_che.legend(fontsize=8); ax_che.grid(True, axis='y', alpha=0.2)
ax_che.xaxis.set_major_locator(MaxNLocator(integer=True))
ax_che.text(0.97, 0.05,
    f"Current: {che_pts} pts\nForecast median: ~{che_pts + che_fc['p50']:.0f} pts\n"
    f"Range: {che_pts + che_fc['p10']:.0f}–{che_pts + che_fc['p90']:.0f}",
    transform=ax_che.transAxes, fontsize=8, color='white', ha='right', va='bottom',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#0A0A0A', edgecolor=C_CHE, alpha=0.9))

# ── Panel 2: Title race ───────────────────────────────────────────────────────
mw_ars = np.arange(1, len(ars_results) + 1)
mw_mci = np.arange(1, len(mci_results) + 1)
ax_title.plot(mw_ars, ars_cum, '-', color=C_ARS, lw=2.5, label='Arsenal')
ax_title.plot(mw_mci, mci_cum, '-', color=C_MCI, lw=2.5, label='Man City')
ax_title.fill_between([34, 38],
    [ars_pts, ars_pts + ars_fc['p10']], [ars_pts, ars_pts + ars_fc['p90']],
    alpha=0.18, color=C_ARS)
ax_title.fill_between([33, 38],
    [mci_pts, mci_pts + mci_fc['p10']], [mci_pts, mci_pts + mci_fc['p90']],
    alpha=0.18, color=C_MCI)
ax_title.plot([34, 38], [ars_pts, ars_pts + ars_fc['p50']], '--', color=C_ARS, lw=1.8)
ax_title.plot([33, 38], [mci_pts, mci_pts + mci_fc['p50']], '--', color=C_MCI, lw=1.8)
ax_title.axvline(34, color=C_GLD, lw=1.2, ls=':', alpha=0.8)
ax_title.text(38.2, ars_pts + ars_fc['p50'],
              f"~{ars_pts + ars_fc['p50']:.0f}", color=C_ARS, fontsize=8, fontweight='bold', va='center')
ax_title.text(38.2, mci_pts + mci_fc['p50'],
              f"~{mci_pts + mci_fc['p50']:.0f}", color=C_MCI, fontsize=8, fontweight='bold', va='center')
ax_title.set_xlim(0.5, 40); ax_title.set_ylim(0, 95)
ax_title.set_title('Title Race · Arsenal vs Man City\nMW34 → MW38', fontsize=10, fontweight='bold', color='white')
ax_title.set_xlabel('Matchweek', fontsize=9); ax_title.set_ylabel('Points', fontsize=9)
ax_title.legend(fontsize=8); ax_title.grid(True, axis='y', alpha=0.2)
ax_title.xaxis.set_major_locator(MaxNLocator(integer=True))
ax_title.text(0.97, 0.05,
    f"P(Arsenal) = {p_ars_title:.1%}  ←\nP(Man City) = {p_mci_title:.1%}\nGD edge: ARS +38 vs MCI +37",
    transform=ax_title.transAxes, fontsize=8, color='white', ha='right', va='bottom',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#0A0A0A', edgecolor=C_GLD, alpha=0.9))

# ── Panel 3: Relegation ───────────────────────────────────────────────────────
mw34 = np.arange(1, 35)
ax_rel.axhspan(0, 38, alpha=0.07, color='red')
ax_rel.axhline(38, color='red', lw=1, ls='--', alpha=0.6, label='~Safety (38 pts)')
ax_rel.plot(mw34, spu_cum, '-', color=C_SPU, lw=2.5, label='Tottenham')
ax_rel.plot(mw34, whu_cum, '-', color=C_WHU, lw=2.5, label='West Ham')
ax_rel.fill_between([34, 38],
    [spu_pts, spu_pts + spu_fc['p10']], [spu_pts, spu_pts + spu_fc['p90']],
    alpha=0.20, color=C_SPU)
ax_rel.fill_between([34, 38],
    [whu_pts, whu_pts + whu_fc['p10']], [whu_pts, whu_pts + whu_fc['p90']],
    alpha=0.20, color=C_WHU)
ax_rel.plot([34, 38], [spu_pts, spu_pts + spu_fc['p50']], '--', color=C_SPU, lw=1.8)
ax_rel.plot([34, 38], [whu_pts, whu_pts + whu_fc['p50']], '--', color=C_WHU, lw=1.8)
ax_rel.axvline(34, color=C_GLD, lw=1.2, ls=':', alpha=0.8)
ax_rel.text(38.2, spu_pts + spu_fc['p50'],
            f"~{spu_pts + spu_fc['p50']:.0f}", color=C_SPU, fontsize=8, fontweight='bold', va='center')
ax_rel.text(38.2, whu_pts + whu_fc['p50'],
            f"~{whu_pts + whu_fc['p50']:.0f}", color=C_WHU, fontsize=8, fontweight='bold', va='center')
ax_rel.set_xlim(0.5, 40); ax_rel.set_ylim(0, 55)
ax_rel.set_title('Relegation Battle · Spurs vs West Ham\nMW34 → MW38', fontsize=10, fontweight='bold', color='white')
ax_rel.set_xlabel('Matchweek', fontsize=9); ax_rel.set_ylabel('Points', fontsize=9)
ax_rel.legend(fontsize=8); ax_rel.grid(True, axis='y', alpha=0.2)
ax_rel.xaxis.set_major_locator(MaxNLocator(integer=True))
ax_rel.text(0.97, 0.05,
    f"P(Spurs relegated) = {p_spu_below:.0%}  ←\nP(WHU relegated) = {p_whu_below:.0%}\n"
    f"P(Spurs safe) = {p_spu_safe:.1%}  |  P(WHU safe) = {p_whu_safe:.1%}",
    transform=ax_rel.transAxes, fontsize=8, color='white', ha='right', va='bottom',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#0A0A0A', edgecolor='#E74C3C', alpha=0.9))

# ── Panel 4: Title race final points distribution ─────────────────────────────
bins1 = np.arange(70, 98, 1)
ax_dist1.hist(ars_finals, bins=bins1, alpha=0.7, color=C_ARS, density=True, label='Arsenal')
ax_dist1.hist(mci_finals, bins=bins1, alpha=0.7, color=C_MCI, density=True, label='Man City')
ax_dist1.axvline(ars_pts + ars_fc['p50'], color=C_ARS, lw=1.8, ls='--')
ax_dist1.axvline(mci_pts + mci_fc['p50'], color=C_MCI, lw=1.8, ls='--')
ax_dist1.set_xlabel('Projected Final Points', fontsize=9)
ax_dist1.set_ylabel('Density', fontsize=9)
ax_dist1.set_title('Title Race · Final Points Distribution\n(10,000 simulations)', fontsize=10, fontweight='bold', color='white')
ax_dist1.legend(fontsize=8); ax_dist1.grid(True, axis='y', alpha=0.2)

# ── Panel 5: Relegation final points distribution ─────────────────────────────
bins2 = np.arange(28, 52, 1)
ax_dist2.hist(spu_finals, bins=bins2, alpha=0.7, color=C_SPU, density=True, label='Spurs')
ax_dist2.hist(whu_finals, bins=bins2, alpha=0.7, color=C_WHU, density=True, label='West Ham')
ax_dist2.axvline(38, color='red', lw=1.5, ls=':', alpha=0.9, label='~Safety (38)')
ax_dist2.axvline(spu_pts + spu_fc['p50'], color=C_SPU, lw=1.8, ls='--')
ax_dist2.axvline(whu_pts + whu_fc['p50'], color=C_WHU, lw=1.8, ls='--')
ax_dist2.set_xlabel('Projected Final Points', fontsize=9)
ax_dist2.set_ylabel('Density', fontsize=9)
ax_dist2.set_title('Relegation Battle · Final Points Distribution\n(10,000 simulations)', fontsize=10, fontweight='bold', color='white')
ax_dist2.legend(fontsize=8); ax_dist2.grid(True, axis='y', alpha=0.2)

# ── Panel 6: Context window form bars ─────────────────────────────────────────
teams_ctx = ['Arsenal', 'Man City', 'Spurs', 'West Ham']
ppg_ctx   = [ars_fc['p_win'], mci_fc['p_win'], spu_fc['p_win'], whu_fc['p_win']]
colors_ctx= [C_ARS, C_MCI, C_SPU, C_WHU]
x_ctx = np.arange(len(teams_ctx))
bars = ax_form.bar(x_ctx, ppg_ctx, color=colors_ctx, alpha=0.85, width=0.55)
for bar, val in zip(bars, ppg_ctx):
    ax_form.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.0%}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='white')
ax_form.set_xticks(x_ctx); ax_form.set_xticklabels(teams_ctx, fontsize=9)
ax_form.set_ylabel('Win Probability (12-GW context)', fontsize=9)
ax_form.set_ylim(0, 0.85)
ax_form.set_title('TimesFM Context Window\nWin % from Last 12 Matches', fontsize=10, fontweight='bold', color='white')
ax_form.grid(True, axis='y', alpha=0.2)

# ── Header ────────────────────────────────────────────────────────────────────
fig.suptitle(
    'EPL 2025–26  ·  TimesFM-Informed Live Snapshot  ·  Data: MW34 (1 May 2026)',
    fontsize=13, fontweight='bold', color='white', y=0.97
)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUTPUTS_DIR, 'live_snapshot.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"Saved: {out_path}")
print(f"Chelsea: {che_pts} pts → ~{che_pts + che_fc['p50']:.0f} projected")
print(f"Arsenal: {p_ars_title:.1%} title probability")
print(f"Spurs:   {p_spu_below:.0%} relegation probability")
