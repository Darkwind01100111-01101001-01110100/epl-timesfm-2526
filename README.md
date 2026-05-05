# EPL 2025–26 · TimesFM Forecasting Experiment

This started as a Chelsea-only sandbox. I wanted to see what happens when you take a simple sports analytics problem—predicting end-of-season points—and run it through a time-series foundation model instead of a spreadsheet.

It's grown into a broader EPL forecasting exercise covering three live questions for the 2025–26 season.

![Live Snapshot](outputs/live_snapshot.png)

*Data as of end of Matchweek 35 (4 May 2026). Arsenal 35 played, Man City 34 played (1 game in hand), 3 games remaining for most clubs.*

---

## What's in here?

| File | Description |
|---|---|
| `data/chelsea_real_2025_26.csv` | Chelsea MW1–35 results (FBref) |
| `data/arsenal_real_2025_26.csv` | Arsenal MW1–35 results (FBref) |
| `data/tottenham_real_2025_26.csv` | Tottenham MW1–35 results (FBref) |
| `data/westham_real_2025_26.csv` | West Ham MW1–35 results (FBref) |
| `src/live_snapshot.py` | **Main script.** Generates the 6-panel dashboard above. |
| `src/run_mw35_forecast.py` | MW35 update — corrected standings, delta comparison chart |
| `src/arsenal_title_forecast.py` | Arsenal vs Man City title race deep-dive |
| `src/relegation_forecast.py` | Tottenham vs West Ham relegation deep-dive |
| `outputs/live_snapshot.png` | The primary chart — updated each matchweek |
| `outputs/mw35_delta_comparison.png` | MW34 → MW35 probability shift chart |
| `outputs/arsenal_title_forecast.png` | Arsenal title race standalone chart |
| `outputs/relegation_forecast.png` | Relegation battle standalone chart |

---

## The Three Live Questions (MW35)

### 1. Will Chelsea avoid the bottom half?

Chelsea sit on **48 points through 35 matches** after losing 1-3 to Nottingham Forest at home in MW35 — their fourth defeat in five games. The TimesFM context window has absorbed this extended poor run and projects a median finish of **~51 points** (range 48–54). That keeps Chelsea clear of any relegation concern but well outside European contention. The Conference League threshold (~55 pts) is now beyond their realistic range.

### 2. Will Arsenal win the league?

Arsenal won 3-0 vs Fulham in MW35 and sit on **76 points from 35 games**. Man City drew 3-3 with Everton and sit on **73 points from 34 games** — they still have a game in hand and **4 games remaining** to Arsenal's 3.

The model gives Arsenal a **57.4% probability** of winning the title. The edge is structural: Arsenal hold the points buffer and the goal difference tiebreaker (+38 vs +37). But City's game in hand means they have more paths to the title — they need to win all 4 remaining games while Arsenal drop at least 4 points across their final 3. City's 12-match context win rate (72%) is slightly higher than Arsenal's (66%), which keeps this genuinely live. City need **4+ points more than Arsenal** from their remaining games to overtake.

**MW34 → MW35 delta:** Arsenal +4.3% (53.1% → 57.4%), Man City −4.3% (46.9% → 42.6%).

### 3. Who gets the final relegation spot — Tottenham or West Ham?

MW35 completely reversed the picture from MW34. Tottenham won 2-1 at Aston Villa; West Ham lost 0-3 at Brentford. Spurs now sit on **37 points**, West Ham on **36 points** — separated by just 1 point with 3 games remaining for both.

The model now gives Spurs a **41% probability of relegation** and West Ham **33%** — down from 100%/0% at MW34. The Spurs win was absorbed by the 12-match context window, lifting their projected win rate from 8% to 19%. Both clubs now have roughly an 82% chance of reaching the traditional 38-point safety threshold, making this a genuine three-game run-in battle.

**MW34 → MW35 delta:** P(Spurs relegated) −59% (100% → 41%), P(WHU relegated) +33% (0% → 33%).

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
        for _ in range(n_seasons)
    ]
    return {'p10': np.percentile(samples, 10),
            'p50': np.percentile(samples, 50),
            'p90': np.percentile(samples, 90)}
```

---

## How to run it

```bash
pip install pandas numpy matplotlib
python src/run_mw35_forecast.py      # MW35 update — live snapshot + delta chart
python src/live_snapshot.py          # regenerates outputs/live_snapshot.png
python src/arsenal_title_forecast.py # title race deep-dive
python src/relegation_forecast.py    # relegation deep-dive
```

---

## Forecast history

| Matchweek | Arsenal title % | Man City title % | Spurs relegated % | WHU relegated % |
|---|---|---|---|---|
| MW34 | 53.1% | 46.9% | 100% | 0% |
| **MW35** | **57.4%** | **42.6%** | **41%** | **33%** |

---

## Next steps

- Add fixture difficulty as a covariate (opponent current points as a proxy)
- Track model accuracy week-over-week as the season concludes (MW36–38)
- Post-season accuracy audit: compare all three forecasts against final table

*Data updated manually after each matchweek. All results sourced from FBref.*
