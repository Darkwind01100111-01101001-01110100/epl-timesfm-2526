# Chelsea FC vs. TimesFM: A Forecasting Experiment

This is an exploratory sandbox. I wanted to see what happens when you take a simple sports analytics problem—predicting end-of-season points—and run it through a massive AI foundation model instead of a spreadsheet.

Currently, Chelsea has 48 points through 31 matches. Simple linear math says they finish with 59 points. I wanted to see if Google's **TimesFM 2.5** (a zero-shot time-series model) could read the context of their recent flat form and brutal remaining schedule better than a simple average.

*Spoiler: It does. It projects 56 points. This repo tracks that tension.*

![Live Snapshot](outputs/live_snapshot.png)

---

## What's actually in here?

This isn't an expert machine learning showcase. It's a practical application of a very cool new tool.

* **`data/chelsea_real_2025_26.csv`**: Real match-by-match results scraped from FBref. Updated manually after each matchweek.
* **`src/live_snapshot.py`**: The core script. It loads the real data, runs the TimesFM forecast on the per-match points, and generates the dashboard visualization above.
* **`src/carousel.py`**: Generates the 4-slide square carousel used for sharing updates on LinkedIn/Threads.
* **`TIMESFM_LIVE_SNAPSHOT.md`**: A clean, shareable markdown document explaining the methodology and the "why."
* **`MODEL_GAPS.md`**: My notes on where the model is currently blind (e.g., player injuries) and how I plan to patch those gaps next.

---

## The Core Insight: Framing the Data

The biggest learning wasn't installing the model; it was figuring out how to ask it the right question.

If you ask TimesFM to forecast a *cumulative* points line that ends in three straight losses (flat line), it just predicts that the line stays flat forever. It predicts the "level," not the growth.

The fix is to forecast the *per-match points* (0, 1, or 3) and then accumulate those on top of the current baseline. Here is the core code that makes it work:

```python
import timesfm
from timesfm.timesfm_2p5 import timesfm_2p5_torch
import numpy as np

# 1. Initialize the model
model = timesfm_2p5_torch.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)
model.compile(timesfm.ForecastConfig(
    max_context=512, max_horizon=10,
    normalize_inputs=True, use_continuous_quantile_head=True,
    force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True,
))

# 2. Forecast the per-match points (0, 1, or 3) for the next 7 matches
pt_fc, qt_fc = model.forecast(
    horizon=7,
    inputs=[df['points_earned'].values.astype(float)]
)

# 3. Accumulate on top of the current 48 points
per_match_mean = pt_fc[0]
projected_cumulative = 48 + np.cumsum(per_match_mean)

print(f"Projected final points: {int(round(projected_cumulative[-1]))}")
```

---

## How to run it yourself

If you want to pull this down and tinker with it:

1. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn torch
   ```

2. **Install TimesFM:**
   *(Note: 2.5 currently requires installation from source)*
   ```bash
   git clone https://github.com/google-research/timesfm.git
   cd timesfm
   pip install -e .
   ```

3. **Run the visualizer:**
   ```bash
   cd src
   python live_snapshot.py
   ```

## Next Steps

I'm tracking the gap between the simple math (59 pts) and the TimesFM forecast (56 pts) week over week through May. I also plan to introduce a "fixture difficulty" covariate so the model knows the difference between playing Man City and playing Sunderland.

*Data as of end of Matchweek 31 (April 2026).*
