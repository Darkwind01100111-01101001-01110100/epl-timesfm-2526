"""
run_real_forecast.py
---------------------
Loads real Chelsea 2025-26 Premier League match data (MW1-31),
runs TimesFM 2.5 to forecast the remaining 7 matches (MW32-38),
and produces two visualizations:
  1. Cumulative points trajectory with confidence bands
  2. Per-match points earned with rolling form line
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import timesfm
from timesfm.timesfm_2p5 import timesfm_2p5_torch

# ── 1. Load Real Data ─────────────────────────────────────────────────────────
df = pd.read_csv('../data/chelsea_real_2025_26.csv', parse_dates=['date'])
df = df.sort_values('matchweek').reset_index(drop=True)

# Derive points and cumulative points
points_map = {'W': 3, 'D': 1, 'L': 0}
df['points_earned'] = df['result'].map(points_map)
df['cumulative_points'] = df['points_earned'].cumsum()
df['rolling_ppg'] = df['points_earned'].rolling(window=5, min_periods=1).mean()

# Flag congested matches (days_rest < 5)
df['is_congested'] = df['days_rest'] < 5

print(f"Loaded {len(df)} real matchweeks.")
print(f"Record: {df['result'].value_counts().to_dict()}")
print(f"Total points: {df['cumulative_points'].iloc[-1]}")
print(f"Goals for: {df['gf'].sum()} | Goals against: {df['ga'].sum()}")

# ── 2. Load TimesFM ───────────────────────────────────────────────────────────
print("\nLoading TimesFM 2.5...")
model = timesfm_2p5_torch.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)
model.compile(
    timesfm.ForecastConfig(
        max_context=512,
        max_horizon=10,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)
print("Model ready.")

# ── 3. Forecast Cumulative Points ─────────────────────────────────────────────
HORIZON = 7  # Remaining matchweeks (32-38)

point_forecast, quantile_forecast = model.forecast(
    horizon=HORIZON,
    inputs=[df['cumulative_points'].values.astype(float)]
)

mean_pts   = point_forecast[0]
lower_pts  = quantile_forecast[0, :, 0]   # 10th percentile
upper_pts  = quantile_forecast[0, :, -1]  # 90th percentile

future_mw = list(range(32, 32 + HORIZON))
forecast_df = pd.DataFrame({
    'matchweek': future_mw,
    'forecast_mean': mean_pts,
    'forecast_lower_10': lower_pts,
    'forecast_upper_90': upper_pts,
})
forecast_df.to_csv('../outputs/real_points_forecast.csv', index=False)
print(f"\nForecast (MW32-38):")
print(forecast_df.to_string(index=False))

# ── 4. Visualization 1: Cumulative Points Trajectory ─────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
sns.set_style("whitegrid")

# Historical
ax.plot(df['matchweek'], df['cumulative_points'],
        marker='o', color='#034694', linewidth=2.5, label='Actual Points (MW1-31)', zorder=3)

# Connector
ax.plot([df['matchweek'].iloc[-1], future_mw[0]],
        [df['cumulative_points'].iloc[-1], mean_pts[0]],
        color='#FFA500', linewidth=1.5, linestyle='--')

# Forecast
ax.plot(future_mw, mean_pts,
        marker='o', color='#FFA500', linewidth=2.5, linestyle='--',
        label='TimesFM Forecast (Mean)', zorder=3)
ax.fill_between(future_mw, lower_pts, upper_pts,
                color='#FFA500', alpha=0.2, label='80% Confidence Interval')

# Reference lines
ax.axhline(y=60, color='#888', linestyle=':', linewidth=1.2, label='Europa League threshold (~60 pts)')
ax.axhline(y=72, color='#aaa', linestyle=':', linewidth=1.0, label='Top 4 threshold (~72 pts)')

# Annotations
last_mw  = df['matchweek'].iloc[-1]
last_pts = df['cumulative_points'].iloc[-1]
proj_pts = int(round(mean_pts[-1]))
ax.annotate(f'MW{last_mw}: {last_pts} pts',
            xy=(last_mw, last_pts), xytext=(last_mw - 7, last_pts + 4),
            arrowprops=dict(arrowstyle='->', color='#034694'), fontsize=9, color='#034694')
ax.annotate(f'Projected end: ~{proj_pts} pts\n(range {int(lower_pts[-1])}–{int(upper_pts[-1])})',
            xy=(future_mw[-1], mean_pts[-1]),
            xytext=(future_mw[-1] - 8, mean_pts[-1] - 9),
            arrowprops=dict(arrowstyle='->', color='#FFA500'), fontsize=9, color='#8B6914')

ax.set_title('Chelsea FC 2025-26 · Cumulative Points Trajectory\nTimesFM 2.5 Forecast · MW32–38', fontsize=14, pad=12)
ax.set_xlabel('Matchweek', fontsize=11)
ax.set_ylabel('Cumulative Points', fontsize=11)
ax.set_xlim(0, 39)
ax.set_xticks(range(0, 40, 2))
ax.legend(loc='upper left', fontsize=9)

# Data note
fig.text(0.99, 0.01, 'Data: FBref · Chelsea 2025-26 PL · MW1-31 · Apr 2026',
         ha='right', va='bottom', fontsize=7, color='gray')

plt.tight_layout()
plt.savefig('../outputs/real_points_trajectory.png', dpi=300, bbox_inches='tight')
print("\nSaved: outputs/real_points_trajectory.png")

# ── 5. Visualization 2: Per-Match Points + Rolling Form ───────────────────────
fig2, ax2 = plt.subplots(figsize=(13, 5))

# Bar chart of per-match points
colors = {'W': '#034694', 'D': '#aaaaaa', 'L': '#cc2200'}
bar_colors = [colors[r] for r in df['result']]
ax2.bar(df['matchweek'], df['points_earned'], color=bar_colors, width=0.7, alpha=0.85, zorder=2)

# Rolling PPG line
ax2.plot(df['matchweek'], df['rolling_ppg'],
         color='#FFA500', linewidth=2.5, label='Rolling 5-match PPG', zorder=3)

# Mark congested matches
cong = df[df['is_congested']]
ax2.scatter(cong['matchweek'], cong['points_earned'] + 0.05,
            marker='v', color='red', s=50, zorder=4, label='Congested (<5 days rest)')

# Legend patches
w_patch = mpatches.Patch(color='#034694', label='Win (3 pts)')
d_patch = mpatches.Patch(color='#aaaaaa', label='Draw (1 pt)')
l_patch = mpatches.Patch(color='#cc2200', label='Loss (0 pts)')
ax2.legend(handles=[w_patch, d_patch, l_patch,
                    plt.Line2D([0],[0], color='#FFA500', linewidth=2, label='Rolling 5-match PPG'),
                    plt.scatter([], [], marker='v', color='red', s=50, label='Congested match')],
           loc='upper right', fontsize=8)

# Opponent labels on x-axis
ax2.set_xticks(df['matchweek'])
ax2.set_xticklabels(
    [f"MW{row.matchweek}\n{row.opponent[:3].upper()}" for _, row in df.iterrows()],
    fontsize=6.5, rotation=45, ha='right'
)
ax2.set_yticks([0, 1, 3])
ax2.set_yticklabels(['0 (L)', '1 (D)', '3 (W)'])
ax2.set_title('Chelsea FC 2025-26 · Per-Match Points & Rolling Form\nData: FBref · MW1-31 · Apr 2026', fontsize=13, pad=10)
ax2.set_ylabel('Points Earned', fontsize=10)
ax2.set_ylim(-0.2, 3.8)

plt.tight_layout()
plt.savefig('../outputs/real_per_match_form.png', dpi=300, bbox_inches='tight')
print("Saved: outputs/real_per_match_form.png")
