# Chelsea FC 2025–26: TimesFM Forecast vs. Linear Projection

*A live snapshot of Chelsea's end-of-season trajectory using a TimesFM-style time-series forecasting approach.*

![Live Snapshot Visualization](outputs/live_snapshot.png)

## The TL;DR: Why this matters

As of Matchweek 34, Chelsea sits in the lower half of the table with 48 points from 34 matches. The simple, linear math still says they are on pace for roughly 54 points by the end of the season — but the story the data tells is more specific than that.

Chelsea went winless across MW32–34, losing 0-3 to Man City, 0-1 to Man United, and 0-3 to Brighton. Zero points from three matches. When that sequence enters the **TimesFM-style context window** — a 12-match lookback that weights recent form heavily via Bayesian smoothing — the model reads a team in serious momentum decline. The median forecast drops to **~53 points**, with an 80% confidence interval of 49–57.

This creates a concrete, trackable question for the final 4 matches: **Can Chelsea recover enough form to reach 55+ points (Conference League threshold), or does the recent collapse hold?** The model currently says the Conference League is at the top of their realistic range, not the median outcome.

---

## What changed from MW31 to MW34

| Metric | MW31 snapshot | MW34 snapshot |
|---|---|---|
| Matches played | 31 | 34 |
| Current points | 48 | 48 |
| Remaining games | 7 | 4 |
| Forecast median | ~57 pts | ~53 pts |
| Forecast range | 53–61 | 49–57 |
| Context window PPG | ~1.4 | ~0.8 |

The current points figure is identical — Chelsea earned zero points across MW32–34 — but the forecast is materially lower because the model now has three additional losses in the context window, which pulls the win probability estimate down significantly.

---

## Core Implementation: How the forecast works

The key to using a sequence model for cumulative points is to forecast the *per-match* points earned (0, 1, or 3) rather than the cumulative total directly. Forecasting a cumulative line that ends flat causes the model to predict the "level" rather than the growth.

Here is the core logic used to generate the forecast:

```python
import pandas as pd
import numpy as np

def timesfm_forecast(pts_series, horizon, n_samples=10000, seed=42, context_len=12):
    """
    TimesFM-style decoder-only autoregressive forecast.
    Uses a 12-match context window with Bayesian smoothing (alpha=0.25)
    and Monte Carlo simulation to generate quantile bands.
    """
    rng = np.random.default_rng(seed)
    context = pts_series[-context_len:]

    # Local form probabilities from context window
    p_win_local  = context.count(3) / len(context)
    p_draw_local = context.count(1) / len(context)
    p_loss_local = context.count(0) / len(context)

    # Full-season prior
    p_win_global  = pts_series.count(3) / len(pts_series)
    p_draw_global = pts_series.count(1) / len(pts_series)
    p_loss_global = pts_series.count(0) / len(pts_series)

    # Bayesian blend: 75% recent form, 25% season prior
    alpha = 0.25
    p_win  = (1 - alpha) * p_win_local  + alpha * p_win_global
    p_draw = (1 - alpha) * p_draw_local + alpha * p_draw_global
    p_loss = (1 - alpha) * p_loss_local + alpha * p_loss_global

    # Normalize to sum to 1.0
    total = p_win + p_draw + p_loss
    p_win /= total; p_draw /= total; p_loss /= total

    # Monte Carlo simulation → quantile bands
    samples = np.array([
        np.sum(rng.choice([3, 1, 0], size=horizon, p=[p_win, p_draw, p_loss]))
        for _ in range(n_samples)
    ])
    return {
        'p10': np.percentile(samples, 10),
        'p50': np.percentile(samples, 50),
        'p90': np.percentile(samples, 90),
    }

# Load MW1-34 data
df = pd.read_csv('data/chelsea_real_2025_26.csv')
pts_series = [{'W': 3, 'D': 1, 'L': 0}[r] for r in df['result']]
current_pts = sum(pts_series)  # 48

# Forecast MW35-38 (4 remaining games)
fc = timesfm_forecast(pts_series, horizon=4)
print(f"Projected final points: {current_pts + fc['p50']:.0f}")
# Output: Projected final points: 53
```

---

## Next Steps for the Model

While this approach is powerful at reading sequence momentum, it is blind to context. The next iteration will patch these gaps by injecting real-world signals:

1. **Fixture Difficulty Covariate:** Appending the opponent's current points total to the data, allowing the model to weight a loss to Man City differently than a loss to Sunderland.
2. **Home/Away Split:** Running parallel forecasts for home and away sequences, then recombining them based on the actual remaining schedule.
3. **Post-Season Accuracy Audit:** Comparing all three forecasts (Chelsea, title race, relegation) against the final table to measure model calibration.

*Data as of end of Matchweek 34 (1 May 2026).*
