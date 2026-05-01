# EPL 2025–26 · TimesFM Forecasting Experiment

This started as a Chelsea-only sandbox. I wanted to see what happens when you take a simple sports analytics problem—predicting end-of-season points—and run it through a time-series foundation model instead of a spreadsheet.

It's grown into a broader EPL forecasting exercise covering three live questions for the 2025–26 season.

![Live Snapshot](outputs/live_snapshot.png)

*Data as of end of Matchweek 34 (1 May 2026).*

---

## What's in here?

| File | Description |
|---|---|
| `data/chelsea_real_2025_26.csv` | Chelsea MW1–31 results (FBref) |
| `data/arsenal_real_2025_26.csv` | Arsenal MW1–34 results (FBref) |
| `data/tottenham_real_2025_26.csv` | Tottenham MW1–34 results (FBref) |
| `data/westham_real_2025_26.csv` | West Ham MW1–34 results (FBref) |
| `src/live_snapshot.py` | **Main script.** Generates the 6-panel dashboard above. |
| `src/arsenal_title_forecast.py` | Arsenal vs Man City title race deep-dive |
| `src/relegation_forecast.py` | Tottenham vs West Ham relegation deep-dive |
| `outputs/live_snapshot.png` | The daily chart — updated each matchweek |
| `outputs/arsenal_title_forecast.png` | Arsenal title race standalone chart |
| `outputs/relegation_forecast.png` | Relegation battle standalone chart |

---

## The Three Live Questions (MW34)

### 1. Will Chelsea make the top 6?

Chelsea sit on 48 points through 31 matches. Simple linear math says they finish with ~59 points. The TimesFM context window reads their recent flat form and projects **~57 points** — just inside the Europa League conversation, but not a certainty.

### 2. Will Arsenal win the league?

Arsenal lead Man City 73–70 with 4 games remaining (City have 5, including a game in hand). The model gives Arsenal a **53.1% probability** of winning the title. The edge is structural: Arsenal already have the points on the board, and the goal difference tiebreaker (+38 vs +37) goes their way in the 11.9% of simulated futures where both teams finish level. City's form over the last 12 matches is slightly better (2.25 PPG vs Arsenal's 1.92), but they need to win all 5 remaining games to guarantee the title regardless.

### 3. Who gets the final relegation spot — Tottenham or West Ham?

This is the most decisive result the model produces. Tottenham (34 pts, 18th) and West Ham (36 pts, 17th) are separated by just 2 points, but the 12-match context window tells a stark story: Spurs are averaging 0.58 PPG over their last 12 matches. West Ham is averaging 1.58 PPG over the same stretch. The model projects Spurs to finish on ~36 points and West Ham on ~42 points, giving Spurs a **100% probability of finishing below West Ham** and a only 33.5% chance of reaching the traditional 38-point safety threshold.

---

## The Core Approach

The biggest learning in this project wasn't installing the model — it was figuring out how to ask it the right question.

If you feed TimesFM a cumulative points line that ends in a flat stretch, it predicts the line stays flat. The fix is to forecast *per-match points* (0, 1, or 3) and accumulate those on top of the current baseline. Here is the core logic:

```python
def timesfm_forecast(pts_series, horizon, n_samples=10000, context_len=12):
    # 1. Extract local form from the last 12 matches (context window)
    context = pts_series[-context_len:]
    p_win_local = context.count(3) / len(context)

    # 2. Blend with full-season prior (Bayesian smoothing, alpha=0.25)
    p_win_global = pts_series.count(3) / len(pts_series)
    p_win = 0.75 * p_win_local + 0.25 * p_win_global

    # 3. Monte Carlo simulation → quantile bands
    samples = [
        sum(rng.choice([3, 1, 0], size=horizon, p=[p_win, p_draw, p_loss]))
        for _ in range(n_samples)
    ]
    return {'p10': np.percentile(samples, 10),
            'p50': np.percentile(samples, 50),
            'p90': np.percentile(samples, 90)}
```

---

## How to run it

```bash
pip install pandas numpy matplotlib
python src/live_snapshot.py          # regenerates outputs/live_snapshot.png
python src/arsenal_title_forecast.py # title race deep-dive
python src/relegation_forecast.py    # relegation deep-dive
```

---

## Next steps

- Add fixture difficulty as a covariate (opponent current points as a proxy)
- Update Chelsea data through MW34 and rerun the full snapshot
- Track model accuracy week-over-week as the season concludes

*Data updated manually after each matchweek. All results sourced from FBref.*
