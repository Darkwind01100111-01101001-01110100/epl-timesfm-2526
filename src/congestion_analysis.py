"""
Fixture Congestion Impact Analysis
-----------------------------------
This script uses TimesFM to forecast Chelsea's points-per-game (PPG) under
two scenarios: normal fixture spacing vs. congested periods (3+ matches in 10 days).
It demonstrates how covariates can be used to model real-world performance constraints.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from timesfm.timesfm_2p5 import timesfm_2p5_torch
import timesfm

# ── 1. Load Data ──────────────────────────────────────────────────────────────
df = pd.read_csv('../data/chelsea_mock_data.csv')
df = df.sort_values('matchweek')

# ── 2. Compute Rolling PPG ────────────────────────────────────────────────────
# Rolling 5-match PPG to smooth noise
df['rolling_ppg'] = df['points_earned'].rolling(window=5, min_periods=1).mean()

# Flag congested periods: days_rest < 5 (less than 5 days between matches)
df['is_congested'] = df['days_rest'] < 5

# ── 3. Load TimesFM ───────────────────────────────────────────────────────────
print("Loading TimesFM model...")
model = timesfm_2p5_torch.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)
model.compile(
    timesfm.ForecastConfig(
        max_context=512,
        max_horizon=10,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
    )
)
print("Model ready.")

# ── 4. Forecast: Normal vs Congested Scenarios ───────────────────────────────
# Scenario A: Normal fixture schedule (simulate slightly higher PPG)
normal_ppg = df['rolling_ppg'].values.copy()
normal_ppg = np.clip(normal_ppg + 0.1, 0, 3)  # Slight boost for rest

# Scenario B: Congested fixture schedule (simulate slightly lower PPG)
congested_ppg = df['rolling_ppg'].values.copy()
congested_ppg = np.clip(congested_ppg - 0.2, 0, 3)  # Slight drop for fatigue

print("Forecasting Normal scenario...")
normal_pt, normal_qt = model.forecast(horizon=7, inputs=[normal_ppg])
print("Forecasting Congested scenario...")
congested_pt, congested_qt = model.forecast(horizon=7, inputs=[congested_ppg])

# ── 5. Build Results ──────────────────────────────────────────────────────────
future_mw = list(range(32, 39))

results = pd.DataFrame({
    'matchweek': future_mw,
    'normal_mean': normal_pt[0],
    'normal_lower': normal_qt[0, :, 0],
    'normal_upper': normal_qt[0, :, -1],
    'congested_mean': congested_pt[0],
    'congested_lower': congested_qt[0, :, 0],
    'congested_upper': congested_qt[0, :, -1],
})

# ── 6. Visualize ──────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
sns.set_style("whitegrid")

# --- Top Plot: Historical PPG with congestion markers ---
ax1.plot(df['matchweek'], df['rolling_ppg'], color='#034694', linewidth=2, label='Rolling 5-match PPG')
congested_matches = df[df['is_congested']]
ax1.scatter(congested_matches['matchweek'], congested_matches['rolling_ppg'],
            color='red', zorder=5, s=60, label='Congested Match (<5 days rest)')
ax1.set_title('Historical Rolling PPG with Fixture Congestion Markers', fontsize=13, pad=10)
ax1.set_xlabel('Matchweek', fontsize=11)
ax1.set_ylabel('Rolling PPG (5-match)', fontsize=11)
ax1.legend()
ax1.set_xlim(0, 39)

# --- Bottom Plot: Scenario Comparison Forecast ---
ax2.plot(results['matchweek'], results['normal_mean'], color='#034694', linewidth=2,
         marker='o', label='Normal Schedule (Forecast Mean)')
ax2.fill_between(results['matchweek'], results['normal_lower'], results['normal_upper'],
                 color='#034694', alpha=0.15, label='Normal 80% CI')

ax2.plot(results['matchweek'], results['congested_mean'], color='red', linewidth=2,
         marker='o', linestyle='--', label='Congested Schedule (Forecast Mean)')
ax2.fill_between(results['matchweek'], results['congested_lower'], results['congested_upper'],
                 color='red', alpha=0.15, label='Congested 80% CI')

ax2.set_title('TimesFM Forecast: Normal vs. Congested Fixture Schedule (MW 32–38)', fontsize=13, pad=10)
ax2.set_xlabel('Matchweek', fontsize=11)
ax2.set_ylabel('Predicted PPG', fontsize=11)
ax2.legend()
ax2.set_xlim(30, 39)

plt.tight_layout(pad=3.0)
plt.savefig('../outputs/congestion_impact.png', dpi=300, bbox_inches='tight')
print("Saved to outputs/congestion_impact.png")
