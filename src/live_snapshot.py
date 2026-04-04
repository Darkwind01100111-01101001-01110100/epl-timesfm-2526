"""
live_snapshot.py
-----------------
Single-chart "live snapshot" visualization combining:
  - Actual 2025-26 cumulative points (MW1-31, real data)
  - TimesFM forecast with 80% CI (MW32-38)
  - CL / EL / Conference League qualification thresholds
  - Remaining fixture difficulty overlay
  - Current table context (top-6 positions)

Forecast approach:
  TimesFM forecasts the per-match points series (0/1/3 per game),
  then those are accumulated on top of the current 48-point baseline.
  This avoids the "level prediction" problem that occurs when forecasting
  a cumulative series that ends in a flat run (MW29-31 = 0 pts each).

Data: FBref · Chelsea 2025-26 PL · MW1-31 · Apr 4 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import timesfm
from timesfm.timesfm_2p5 import timesfm_2p5_torch

# ── 1. Real Data ──────────────────────────────────────────────────────────────
df = pd.read_csv('../data/chelsea_real_2025_26.csv', parse_dates=['date'])
df = df.sort_values('matchweek').reset_index(drop=True)
points_map = {'W': 3, 'D': 1, 'L': 0}
df['points_earned'] = df['result'].map(points_map)
df['cumulative_points'] = df['points_earned'].cumsum()
current_pts = int(df['cumulative_points'].iloc[-1])  # 48

# ── 2. Remaining Fixtures (MW32-38) ───────────────────────────────────────────
remaining = pd.DataFrame({
    'matchweek': [32, 33, 34, 35, 36, 37, 38],
    'opponent':  ['Man City', 'Man Utd', 'Brighton', 'Nott Forest', 'Liverpool', 'Spurs', 'Sunderland'],
    'venue':     ['H', 'H', 'A', 'H', 'A', 'H', 'A'],
    'opp_pts':   [61, 55, 43, 32, 49, 30, 43],  # opponent pts at MW31
})

def difficulty(pts):
    if pts > 50: return 'hard'
    if pts > 34: return 'medium'
    return 'soft'
remaining['difficulty'] = remaining['opp_pts'].apply(difficulty)
diff_colors = {'hard': '#cc2200', 'medium': '#FFA500', 'soft': '#2ca02c'}

# ── 3. TimesFM Forecast ───────────────────────────────────────────────────────
# KEY FIX: forecast per-match points (0/1/3 series), then accumulate from baseline.
# Forecasting the cumulative series directly causes the model to predict the
# "level" of the series (~48) rather than the continuation of growth, because
# the last 3 matches were all 0 pts (flat tail = flat forecast).
print("Loading TimesFM 2.5...")
model = timesfm_2p5_torch.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)
model.compile(timesfm.ForecastConfig(
    max_context=512, max_horizon=10,
    normalize_inputs=True, use_continuous_quantile_head=True,
    force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True,
))

# Forecast per-match points
pt_fc, qt_fc = model.forecast(
    horizon=7,
    inputs=[df['points_earned'].values.astype(float)]
)

# Accumulate from current baseline
per_match_mean  = pt_fc[0]
per_match_lower = qt_fc[0, :, 0]
per_match_upper = qt_fc[0, :, -1]

mean_pts  = current_pts + np.cumsum(per_match_mean)
lower_pts = current_pts + np.cumsum(per_match_lower)
upper_pts = current_pts + np.cumsum(per_match_upper)
future_mw = list(range(32, 39))

proj_final_mean  = int(round(mean_pts[-1]))
proj_final_lower = int(round(lower_pts[-1]))
proj_final_upper = int(round(upper_pts[-1]))

print(f"Forecast: {proj_final_mean} pts (range {proj_final_lower}–{proj_final_upper})")

# ── 4. Current Table Context (MW31) ───────────────────────────────────────────
table = pd.DataFrame({
    'pos':  [1, 2, 3, 4, 5, 6],
    'team': ['Arsenal', 'Man City', 'Man Utd', 'Aston Villa', 'Liverpool', 'Chelsea'],
    'mp':   [31, 30, 31, 31, 31, 31],
    'pts':  [70, 61, 55, 54, 49, 48],
    'gd':   [39, 32, 13, 5, 8, 15],
})
table['proj_pts'] = (table['pts'] / table['mp'] * 38).round(0).astype(int)

# ── 5. Qualification Thresholds ───────────────────────────────────────────────
CL_THRESHOLD   = 66   # 4th place (Aston Villa pace)
EL_THRESHOLD   = 60   # 5th place (Liverpool pace)
UECL_THRESHOLD = 55   # 6th place boundary

# ── 6. Build the Figure ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 9))
gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[3, 1], hspace=0.45, wspace=0.35)
ax_main  = fig.add_subplot(gs[0, :])
ax_fix   = fig.add_subplot(gs[1, 0])
ax_table = fig.add_subplot(gs[1, 1])

sns.set_style("whitegrid")

# ── Main Chart ────────────────────────────────────────────────────────────────
# Historical
ax_main.plot(df['matchweek'], df['cumulative_points'],
             color='#034694', linewidth=2.5, marker='o', markersize=4,
             label='Actual Points (MW1–31)', zorder=4)

# Connector from last actual to first forecast
ax_main.plot([31, 32], [current_pts, mean_pts[0]],
             color='#FFA500', linewidth=1.5, linestyle='--', zorder=3)

# Forecast line
ax_main.plot(future_mw, mean_pts,
             color='#FFA500', linewidth=2.5, linestyle='--', marker='o', markersize=5,
             label=f'TimesFM Forecast (mean: ~{proj_final_mean} pts)', zorder=4)
ax_main.fill_between(future_mw, lower_pts, upper_pts,
                     color='#FFA500', alpha=0.18,
                     label=f'80% Confidence Interval ({proj_final_lower}–{proj_final_upper} pts)')

# Threshold bands (shaded zones)
ax_main.axhspan(CL_THRESHOLD, 85, alpha=0.04, color='#1a73e8', zorder=0)
ax_main.axhspan(EL_THRESHOLD, CL_THRESHOLD, alpha=0.06, color='#ff6d00', zorder=0)
ax_main.axhspan(UECL_THRESHOLD, EL_THRESHOLD, alpha=0.06, color='#2ca02c', zorder=0)

ax_main.axhline(CL_THRESHOLD,   color='#1a73e8', linewidth=1.2, linestyle='--', alpha=0.7)
ax_main.axhline(EL_THRESHOLD,   color='#ff6d00', linewidth=1.2, linestyle='--', alpha=0.7)
ax_main.axhline(UECL_THRESHOLD, color='#2ca02c', linewidth=1.2, linestyle='--', alpha=0.7)

ax_main.text(38.3, CL_THRESHOLD + 0.4,   f'CL (~{CL_THRESHOLD})',   color='#1a73e8', fontsize=8, va='bottom')
ax_main.text(38.3, EL_THRESHOLD + 0.4,   f'EL (~{EL_THRESHOLD})',   color='#ff6d00', fontsize=8, va='bottom')
ax_main.text(38.3, UECL_THRESHOLD + 0.4, f'UECL (~{UECL_THRESHOLD})', color='#2ca02c', fontsize=8, va='bottom')

# Current position annotation
ax_main.annotate('Now: 48 pts · 6th',
                 xy=(31, current_pts), xytext=(25.5, current_pts + 7),
                 arrowprops=dict(arrowstyle='->', color='#034694', lw=1.2),
                 fontsize=9, color='#034694', fontweight='bold')

# Forecast end annotation
ax_main.annotate(f'TimesFM: ~{proj_final_mean} pts\n({proj_final_lower}–{proj_final_upper} range)',
                 xy=(38, mean_pts[-1]),
                 xytext=(34, mean_pts[-1] + 4),
                 arrowprops=dict(arrowstyle='->', color='#8B6914', lw=1.2),
                 fontsize=9, color='#8B6914')

# Linear projection marker
linear_proj = round(current_pts / 31 * 38)
ax_main.scatter([38], [linear_proj], color='#888', marker='x', s=70, zorder=5,
                label=f'Linear projection: {linear_proj} pts')
ax_main.text(38.3, linear_proj, f'{linear_proj}', color='#888', fontsize=8, va='center')

ax_main.set_title(
    'Chelsea FC 2025–26 · Season Trajectory & End-of-Season Forecast\n'
    'TimesFM 2.5 (Google) · Real Data: FBref · MW1–31 · Apr 4 2026',
    fontsize=13, pad=10
)
ax_main.set_xlabel('Matchweek', fontsize=10)
ax_main.set_ylabel('Cumulative Points', fontsize=10)
ax_main.set_xlim(0, 40.5)
ax_main.set_ylim(0, 82)
ax_main.set_xticks(range(0, 40, 2))
ax_main.legend(loc='upper left', fontsize=8.5)

# ── Remaining Fixtures ────────────────────────────────────────────────────────
bar_colors = [diff_colors[d] for d in remaining['difficulty']]
ax_fix.barh(remaining['matchweek'], remaining['opp_pts'],
            color=bar_colors, alpha=0.82, height=0.6)
for _, row in remaining.iterrows():
    label = f"{row['opponent']} ({'H' if row['venue']=='H' else 'A'})"
    ax_fix.text(row['opp_pts'] + 0.8, row['matchweek'], label, va='center', fontsize=8)

ax_fix.set_yticks(remaining['matchweek'])
ax_fix.set_yticklabels([f"MW{mw}" for mw in remaining['matchweek']], fontsize=8)
ax_fix.set_xlabel('Opponent Current Pts (difficulty proxy)', fontsize=8)
ax_fix.set_title('Remaining Fixtures · MW32–38', fontsize=10, pad=6)
ax_fix.set_xlim(0, 82)
ax_fix.invert_yaxis()
ax_fix.legend(handles=[
    mpatches.Patch(color='#cc2200', alpha=0.82, label='Hard  (>50 pts)'),
    mpatches.Patch(color='#FFA500', alpha=0.82, label='Medium (35–50)'),
    mpatches.Patch(color='#2ca02c', alpha=0.82, label='Soft   (<35 pts)'),
], fontsize=7, loc='lower right')

# ── Standings Table ───────────────────────────────────────────────────────────
row_colors = ['#e8f0fe']*4 + ['#fff3e0', '#e8f5e9']
col_labels = ['Pos', 'Team', 'Pts', 'GD', 'Proj']
col_widths  = [0.10, 0.38, 0.14, 0.18, 0.20]
cell_data = [
    [r['pos'], r['team'], r['pts'],
     f"+{r['gd']}" if r['gd'] > 0 else str(r['gd']), r['proj_pts']]
    for _, r in table.iterrows()
]
tbl = ax_table.table(cellText=cell_data, colLabels=col_labels,
                     colWidths=col_widths, cellLoc='center',
                     loc='center', bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor('#1a1a2e')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
for i, color in enumerate(row_colors):
    for j in range(len(col_labels)):
        tbl[i+1, j].set_facecolor(color)
        if table.iloc[i]['team'] == 'Chelsea':
            tbl[i+1, j].set_text_props(fontweight='bold')

ax_table.axis('off')
ax_table.set_title('Top 6 · MW31 Standings\n(Proj = linear to 38 games)', fontsize=10, pad=6)
ax_table.text(0.5, -0.05,
    '\u25a0 Blue = CL  \u25a0 Orange = EL  \u25a0 Green = UECL',
    transform=ax_table.transAxes, ha='center', va='top', fontsize=7.5, color='#555')

plt.savefig('../outputs/live_snapshot.png', dpi=300, bbox_inches='tight')
print(f"Saved: outputs/live_snapshot.png  |  Forecast: {proj_final_mean} pts ({proj_final_lower}–{proj_final_upper})")
