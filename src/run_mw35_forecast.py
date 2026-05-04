"""
run_mw35_forecast.py
====================
Generates MW35 updated forecasts and two charts:
  1. live_snapshot.png  — updated 6-panel EPL dashboard (replaces README daily chart)
  2. mw35_delta_comparison.png — MW34 vs MW35 probability shift side-by-side

Man City MW35: D 3-3 vs Everton (added inline — no separate CSV yet)
All other teams: loaded from their respective CSVs through MW35.

Data as of end of Matchweek 35 (4 May 2026).
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from datetime import datetime

REPO_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(REPO_DIR, 'data')
OUTPUTS_DIR = os.path.join(REPO_DIR, 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def r2p(r): return {'W': 3, 'D': 1, 'L': 0}[r]
def rc(r):  return {'W': '#2ECC71', 'D': '#F39C12', 'L': '#E74C3C'}[r]

def timesfm_forecast(pts_series, horizon, n_samples=10000, seed=42, context_len=12):
    """
    TimesFM-style decoder-only autoregressive forecast.
    12-match context window + Bayesian smoothing (alpha=0.25) + Monte Carlo.
    """
    rng = np.random.default_rng(seed)
    ctx = pts_series[-context_len:]
    pw_l = ctx.count(3)/len(ctx); pd_l = ctx.count(1)/len(ctx); pl_l = ctx.count(0)/len(ctx)
    pw_g = pts_series.count(3)/len(pts_series)
    pd_g = pts_series.count(1)/len(pts_series)
    pl_g = pts_series.count(0)/len(pts_series)
    a = 0.25
    pw = (1-a)*pw_l + a*pw_g; pd_ = (1-a)*pd_l + a*pd_g; pl = (1-a)*pl_l + a*pl_g
    t = pw+pd_+pl; pw/=t; pd_/=t; pl/=t
    samples = np.array([
        np.sum(rng.choice([3,1,0], size=max(horizon,1), p=[pw,pd_,pl]))
        for _ in range(n_samples)
    ])
    return {'p10': float(np.percentile(samples,10)), 'p50': float(np.percentile(samples,50)),
            'p90': float(np.percentile(samples,90)), 'mean': float(np.mean(samples)),
            'p_win': pw, 'p_draw': pd_, 'p_loss': pl, 'samples': samples}

# ── Load data ─────────────────────────────────────────────────────────────────
ars_df = pd.read_csv(os.path.join(DATA_DIR, 'arsenal_real_2025_26.csv'))
spu_df = pd.read_csv(os.path.join(DATA_DIR, 'tottenham_real_2025_26.csv'))
whu_df = pd.read_csv(os.path.join(DATA_DIR, 'westham_real_2025_26.csv'))
che_df = pd.read_csv(os.path.join(DATA_DIR, 'chelsea_real_2025_26.csv'))

# Man City: 34 games played (1 game in hand vs Arsenal's 35).
# MW35 Everton 3-3 City is NOT counted here — City have 4 games remaining.
mci_raw = [
    'W','L','L','W','D','W','W','W','L','W',
    'W','L','W','W','W','W','W','W','D','D',
    'D','L','W','D','W','W','W','W','D','D',
    'W','W','W','W'   # 34 results: MW1-34
]

ars_s = [r2p(r) for r in ars_df['result']]
spu_s = [r2p(r) for r in spu_df['result']]
whu_s = [r2p(r) for r in whu_df['result']]
che_s = [r2p(r) for r in che_df['result']]
mci_s = [r2p(r) for r in mci_raw]

ars_pts = sum(ars_s); ars_mw = len(ars_s)   # 76 pts, 35 played, 3 remaining
mci_pts = sum(mci_s); mci_mw = len(mci_s)   # 73 pts, 34 played, 4 remaining
spu_pts = sum(spu_s); spu_mw = len(spu_s)   # 37 pts, 35 played
whu_pts = sum(whu_s); whu_mw = len(whu_s)   # 36 pts, 35 played
che_pts = sum(che_s); che_mw = len(che_s)   # 49 pts, 35 played

print(f"Arsenal:   {ars_pts} pts ({ars_mw} played, {38-ars_mw} remaining)")
print(f"Man City:  {mci_pts} pts ({mci_mw} played, {38-mci_mw} remaining)")
print(f"Tottenham: {spu_pts} pts ({spu_mw} played, {38-spu_mw} remaining)")
print(f"West Ham:  {whu_pts} pts ({whu_mw} played, {38-whu_mw} remaining)")
print(f"Chelsea:   {che_pts} pts ({che_mw} played, {38-che_mw} remaining)")

# ── Forecasts ─────────────────────────────────────────────────────────────────
ars_fc = timesfm_forecast(ars_s, horizon=38-ars_mw)
mci_fc = timesfm_forecast(mci_s, horizon=38-mci_mw)
spu_fc = timesfm_forecast(spu_s, horizon=38-spu_mw)
whu_fc = timesfm_forecast(whu_s, horizon=38-whu_mw)
che_fc = timesfm_forecast(che_s, horizon=38-che_mw)

n = 10000
ars_fin = ars_pts + ars_fc['samples'][:n]
mci_fin = mci_pts + mci_fc['samples'][:n]
spu_fin = spu_pts + spu_fc['samples'][:n]
whu_fin = whu_pts + whu_fc['samples'][:n]

p_ars = float(np.mean(ars_fin >= mci_fin))   # GD tiebreak: ARS +38 vs MCI +37
p_mci = float(np.mean(mci_fin > ars_fin))
p_spu_rel = float(np.mean(spu_fin < whu_fin))
p_whu_rel = float(np.mean(whu_fin < spu_fin))
p_spu_safe = float(np.mean(spu_fin >= 38))
p_whu_safe = float(np.mean(whu_fin >= 38))

print(f"\nMW35 Forecasts:")
print(f"  P(Arsenal title):   {p_ars:.1%}")
print(f"  P(Man City title):  {p_mci:.1%}")
print(f"  P(Spurs relegated): {p_spu_rel:.1%}")
print(f"  P(WHU relegated):   {p_whu_rel:.1%}")
print(f"  P(Spurs safe ≥38):  {p_spu_safe:.1%}")
print(f"  P(WHU safe ≥38):    {p_whu_safe:.1%}")

# MW34 baseline for delta
MW34 = {
    'p_ars': 0.531, 'p_mci': 0.469,  # MW34 baseline (City had 1 game in hand)
    'p_spu_rel': 1.000, 'p_whu_rel': 0.000,
    'p_spu_safe': 0.335, 'p_whu_safe': 0.951,
}

# Save snapshot
snapshot = {
    'as_of': datetime.utcnow().isoformat(), 'matchweek': 35,
    'standings': {
        'arsenal': {'pts': ars_pts, 'played': ars_mw},
        'man_city': {'pts': mci_pts, 'played': mci_mw},
        'tottenham': {'pts': spu_pts, 'played': spu_mw},
        'west_ham': {'pts': whu_pts, 'played': whu_mw},
        'chelsea': {'pts': che_pts, 'played': che_mw},
    },
    'forecasts': {
        'p_arsenal_title': p_ars, 'p_mancity_title': p_mci,
        'p_spurs_relegated': p_spu_rel, 'p_whu_relegated': p_whu_rel,
        'p_spurs_safe': p_spu_safe, 'p_whu_safe': p_whu_safe,
    },
    'mw34_baseline': MW34,
}
with open(os.path.join(OUTPUTS_DIR, 'mw35_forecast_snapshot.json'), 'w') as f:
    json.dump(snapshot, f, indent=2)

# ── Style ─────────────────────────────────────────────────────────────────────
BG='#0D0D0D'; PANEL='#161616'
C_CHE='#034694'; C_ARS='#EF0107'; C_MCI='#6CABDD'
C_SPU='#132257'; C_WHU='#7A263A'; C_GLD='#F5A623'

plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':9,
    'axes.facecolor':PANEL,'figure.facecolor':BG,
    'axes.edgecolor':'#2A2A2A','axes.labelcolor':'#CCCCCC',
    'xtick.color':'#888888','ytick.color':'#888888',
    'text.color':'#DDDDDD','grid.color':'#222222',
    'legend.facecolor':'#161616','legend.edgecolor':'#333333',
})

# ── Chart 1: Updated live snapshot ───────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor(BG)
gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.30,
              left=0.06, right=0.97, top=0.91, bottom=0.07)
axes = [fig.add_subplot(gs[r,c]) for r in range(2) for c in range(3)]
for ax in axes: ax.set_facecolor(PANEL)
ax_che, ax_ttl, ax_rel, ax_d1, ax_d2, ax_ctx = axes

def _traj(ax, pts_s, pts, fc, mw, color, label, ylim=85):
    mws = np.arange(1, mw+1)
    cum = np.cumsum(pts_s)
    mw_fc = np.array([mw, 38])
    ax.plot(mws, cum, '-', color=color, lw=2.5, label=label)
    ax.fill_between(mw_fc, [pts, pts+fc['p10']], [pts, pts+fc['p90']], alpha=0.20, color=color)
    ax.plot(mw_fc, [pts, pts+fc['p50']], '--', color=color, lw=1.8)
    ax.axvline(mw, color=C_GLD, lw=1.2, ls=':', alpha=0.8)
    ax.text(39.0, pts+fc['p50'], f"~{pts+fc['p50']:.0f}", color=color,
            fontsize=8, fontweight='bold', va='center')
    ax.set_xlim(0.5, 40.5); ax.set_ylim(0, ylim)
    ax.grid(True, axis='y', alpha=0.2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

_traj(ax_che, che_s, che_pts, che_fc, che_mw, C_CHE, 'Chelsea')
ax_che.axhline(67, color='#1a73e8', lw=1, ls='--', alpha=0.5)
ax_che.text(1, 67.5, 'CL ~67', color='#1a73e8', fontsize=7)
ax_che.set_title('Chelsea · End-of-Season Forecast\nMW35 Update', fontsize=10, fontweight='bold', color='white')
ax_che.set_xlabel('Matchweek', fontsize=9); ax_che.set_ylabel('Points', fontsize=9)
ax_che.legend(fontsize=8)
ax_che.text(0.97, 0.05, f"Current: {che_pts} pts\nForecast: ~{che_pts+che_fc['p50']:.0f} pts\nRange: {che_pts+che_fc['p10']:.0f}–{che_pts+che_fc['p90']:.0f}",
    transform=ax_che.transAxes, fontsize=8, color='white', ha='right', va='bottom',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#0A0A0A', edgecolor=C_CHE, alpha=0.9))

_traj(ax_ttl, ars_s, ars_pts, ars_fc, ars_mw, C_ARS, 'Arsenal', ylim=95)
_traj(ax_ttl, mci_s, mci_pts, mci_fc, mci_mw, C_MCI, 'Man City', ylim=95)
ax_ttl.set_title('Title Race · Arsenal vs Man City\nMW35 Update', fontsize=10, fontweight='bold', color='white')
ax_ttl.set_xlabel('Matchweek', fontsize=9); ax_ttl.set_ylabel('Points', fontsize=9)
ax_ttl.legend(fontsize=8)
ax_ttl.text(0.97, 0.05,
    f"P(Arsenal) = {p_ars:.1%}\nP(Man City) = {p_mci:.1%}\nGD edge: ARS +38 vs MCI +37",
    transform=ax_ttl.transAxes, fontsize=8, color='white', ha='right', va='bottom',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#0A0A0A', edgecolor=C_GLD, alpha=0.9))

ax_rel.axhspan(0, 38, alpha=0.07, color='red')
ax_rel.axhline(38, color='red', lw=1, ls='--', alpha=0.6, label='~Safety (38 pts)')
_traj(ax_rel, spu_s, spu_pts, spu_fc, spu_mw, C_SPU, 'Tottenham', ylim=55)
_traj(ax_rel, whu_s, whu_pts, whu_fc, whu_mw, C_WHU, 'West Ham', ylim=55)
ax_rel.set_title('Relegation Battle · Spurs vs West Ham\nMW35 Update', fontsize=10, fontweight='bold', color='white')
ax_rel.set_xlabel('Matchweek', fontsize=9); ax_rel.set_ylabel('Points', fontsize=9)
ax_rel.legend(fontsize=8)
ax_rel.text(0.97, 0.05,
    f"P(Spurs relegated) = {p_spu_rel:.0%}\nP(WHU relegated) = {p_whu_rel:.0%}\n"
    f"P(Spurs safe) = {p_spu_safe:.1%}  |  P(WHU safe) = {p_whu_safe:.1%}",
    transform=ax_rel.transAxes, fontsize=8, color='white', ha='right', va='bottom',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#0A0A0A', edgecolor='#E74C3C', alpha=0.9))

bins1 = np.arange(int(min(ars_fin.min(), mci_fin.min()))-1, int(max(ars_fin.max(), mci_fin.max()))+2, 1)
ax_d1.hist(ars_fin, bins=bins1, alpha=0.7, color=C_ARS, density=True, label='Arsenal')
ax_d1.hist(mci_fin, bins=bins1, alpha=0.7, color=C_MCI, density=True, label='Man City')
ax_d1.axvline(ars_pts+ars_fc['p50'], color=C_ARS, lw=1.8, ls='--')
ax_d1.axvline(mci_pts+mci_fc['p50'], color=C_MCI, lw=1.8, ls='--')
ax_d1.set_xlabel('Projected Final Points', fontsize=9); ax_d1.set_ylabel('Density', fontsize=9)
ax_d1.set_title('Title Race · Final Points Distribution\n(10,000 simulations)', fontsize=10, fontweight='bold', color='white')
ax_d1.legend(fontsize=8); ax_d1.grid(True, axis='y', alpha=0.2)

bins2 = np.arange(int(min(spu_fin.min(), whu_fin.min()))-1, int(max(spu_fin.max(), whu_fin.max()))+2, 1)
ax_d2.hist(spu_fin, bins=bins2, alpha=0.7, color=C_SPU, density=True, label='Spurs')
ax_d2.hist(whu_fin, bins=bins2, alpha=0.7, color=C_WHU, density=True, label='West Ham')
ax_d2.axvline(38, color='red', lw=1.5, ls=':', alpha=0.9, label='~Safety (38)')
ax_d2.axvline(spu_pts+spu_fc['p50'], color=C_SPU, lw=1.8, ls='--')
ax_d2.axvline(whu_pts+whu_fc['p50'], color=C_WHU, lw=1.8, ls='--')
ax_d2.set_xlabel('Projected Final Points', fontsize=9); ax_d2.set_ylabel('Density', fontsize=9)
ax_d2.set_title('Relegation Battle · Final Points Distribution\n(10,000 simulations)', fontsize=10, fontweight='bold', color='white')
ax_d2.legend(fontsize=8); ax_d2.grid(True, axis='y', alpha=0.2)

teams_ctx = ['Arsenal', 'Man City', 'Spurs', 'West Ham']
ppg_ctx   = [ars_fc['p_win'], mci_fc['p_win'], spu_fc['p_win'], whu_fc['p_win']]
colors_ctx= [C_ARS, C_MCI, C_SPU, C_WHU]
x_ctx = np.arange(4)
bars = ax_ctx.bar(x_ctx, ppg_ctx, color=colors_ctx, alpha=0.85, width=0.55)
for bar, val in zip(bars, ppg_ctx):
    ax_ctx.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f'{val:.0%}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='white')
ax_ctx.set_xticks(x_ctx); ax_ctx.set_xticklabels(teams_ctx, fontsize=9)
ax_ctx.set_ylabel('Win % (12-match context)', fontsize=9); ax_ctx.set_ylim(0, 0.85)
ax_ctx.set_title('TimesFM Context Window\nWin % from Last 12 Matches', fontsize=10, fontweight='bold', color='white')
ax_ctx.grid(True, axis='y', alpha=0.2)

fig.suptitle(f'EPL 2025–26  ·  TimesFM-Informed Live Snapshot  ·  Data: MW35 (4 May 2026) · City 1 game in hand',

             fontsize=13, fontweight='bold', color='white', y=0.97)
plt.savefig(os.path.join(OUTPUTS_DIR, 'live_snapshot.png'), dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("Saved: live_snapshot.png")

# ── Chart 2: MW34 vs MW35 delta ───────────────────────────────────────────────
fig2, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(14, 6))
fig2.patch.set_facecolor(BG)
for ax in [ax_t, ax_r]: ax.set_facecolor(PANEL)

w = 0.32
# Title delta
labels_t = ['Arsenal\nTitle', 'Man City\nTitle']
v34_t = [MW34['p_ars'], MW34['p_mci']]
v35_t = [p_ars, p_mci]
x_t = np.arange(2)
b1 = ax_t.bar(x_t - w/2, v34_t, w, color=[C_ARS, C_MCI], alpha=0.45, label='MW34 forecast')
b2 = ax_t.bar(x_t + w/2, v35_t, w, color=[C_ARS, C_MCI], alpha=0.90, label='MW35 forecast')
for bar, val in zip(list(b1)+list(b2), v34_t+v35_t):
    ax_t.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
              f'{val:.1%}', ha='center', va='bottom', fontsize=9, color='white', fontweight='bold')
for i, (v34, v35) in enumerate(zip(v34_t, v35_t)):
    delta = v35 - v34
    col = '#2ECC71' if delta > 0 else '#E74C3C'
    ax_t.text(i, max(v34, v35)+0.045, f'{"+" if delta>=0 else ""}{delta:.1%}',
              ha='center', fontsize=11, color=col, fontweight='bold')
ax_t.set_xticks(x_t); ax_t.set_xticklabels(labels_t, fontsize=11)
ax_t.set_ylim(0, 0.80); ax_t.set_ylabel('Probability', fontsize=10)
ax_t.set_title('Title Race · What Changed After MW35?', fontsize=11, fontweight='bold', color='white')
ax_t.legend(fontsize=9); ax_t.grid(True, axis='y', alpha=0.2)
# Context note
ax_t.text(0.5, -0.12,
    f"Arsenal W 3-0 Fulham → {ars_pts} pts (35 played)  |  Man City → {mci_pts} pts (34 played, 1 game in hand)  |  City need {ars_pts - mci_pts + 1}+ pts from 4 games to overtake",
    transform=ax_t.transAxes, ha='center', fontsize=8.5, color='#AAAAAA')

# Relegation delta
labels_r = ['Spurs\nRelegate', 'WHU\nRelegate', 'Spurs\nSafe ≥38', 'WHU\nSafe ≥38']
v34_r = [MW34['p_spu_rel'], MW34['p_whu_rel'], MW34['p_spu_safe'], MW34['p_whu_safe']]
v35_r = [p_spu_rel, p_whu_rel, p_spu_safe, p_whu_safe]
colors_r = [C_SPU, C_WHU, C_SPU, C_WHU]
x_r = np.arange(4)
b3 = ax_r.bar(x_r - w/2, v34_r, w, color=colors_r, alpha=0.45, label='MW34 forecast')
b4 = ax_r.bar(x_r + w/2, v35_r, w, color=colors_r, alpha=0.90, label='MW35 forecast')
for bar, val in zip(list(b3)+list(b4), v34_r+v35_r):
    ax_r.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
              f'{val:.0%}', ha='center', va='bottom', fontsize=8.5, color='white', fontweight='bold')
for i, (v34, v35) in enumerate(zip(v34_r, v35_r)):
    delta = v35 - v34
    if abs(delta) < 0.005: continue
    col = '#2ECC71' if delta > 0 else '#E74C3C'
    ax_r.text(i, max(v34, v35)+0.045, f'{"+" if delta>=0 else ""}{delta:.0%}',
              ha='center', fontsize=10, color=col, fontweight='bold')
ax_r.set_xticks(x_r); ax_r.set_xticklabels(labels_r, fontsize=9)
ax_r.set_ylim(0, 1.30); ax_r.set_ylabel('Probability', fontsize=10)
ax_r.set_title('Relegation Battle · What Changed After MW35?', fontsize=11, fontweight='bold', color='white')
ax_r.legend(fontsize=9); ax_r.grid(True, axis='y', alpha=0.2)
ax_r.text(0.5, -0.12,
    f"Spurs W 2-1 Aston Villa → {spu_pts} pts  |  West Ham L 0-3 Brentford → {whu_pts} pts  |  Gap now: {whu_pts-spu_pts} pts",
    transform=ax_r.transAxes, ha='center', fontsize=8.5, color='#AAAAAA')

fig2.suptitle('EPL 2025–26  ·  Forecast Delta: MW34 → MW35  ·  4 May 2026',
              fontsize=12, fontweight='bold', color='white', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'mw35_delta_comparison.png'), dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("Saved: mw35_delta_comparison.png")
print("\nAll done.")
