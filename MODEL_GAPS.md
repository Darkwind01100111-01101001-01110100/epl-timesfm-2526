# TimesFM Model Gaps & Patching Strategy

This document maps the known limitations of using TimesFM as a pure sequence model for football prediction, and outlines practical, low-complexity approaches to patch each gap with real-world signals.

The philosophy here is **not** to build a complex ML pipeline. It is to understand where the model is blind, and to inject simple, manually-curated or lightly-scraped signals that shift the forecast in a more informed direction.

---

## The Core Limitation

TimesFM reads the **shape of a time series** — the rhythm, trend, and momentum of a sequence of numbers. It does not read context. When you pass in Chelsea's cumulative points array, it sees:

```
[1, 4, 7, 8, 8, 8, 11, 14, 14, 14, 17, 20, 21, 21, 21, 24, 25, 25, 25, 26, 26, 29, 32, 35, 38, 39, 40, 40, 43, 43, 43, 48...]
```

It does not see that MW28 (0 pts, loss at Arsenal) coincided with Cole Palmer being injured. It does not see that MW29 (3 pts, 4-1 win at Aston Villa) was an anomalous result against a depleted opponent. It forecasts based on the **pattern of the sequence alone**.

---

## Gap Map

| # | Gap | What the model misses | Severity | Patch approach |
|---|---|---|---|---|
| 1 | **Player availability** | Key injuries, suspensions, returns | High | Manual injury flag column in data CSV |
| 2 | **Fixture difficulty** | Opponent quality varies wildly | High | Append opponent current points as a covariate |
| 3 | **Home/away split** | Chelsea's home vs. away PPG differs significantly | Medium | Separate forecast series for H and A, then merge |
| 4 | **Competition fatigue** | UCL/FA Cup midweek matches drain the squad | Medium | `days_rest` column already in data; extend with competition type |
| 5 | **Managerial/tactical shifts** | Formation changes, new signings bedding in | Low-Medium | Manual "regime" flag (e.g., `tactical_shift=1`) |
| 6 | **Opponent form** | A "weak" opponent in good form is dangerous | Medium | Append opponent last-5 PPG as covariate |

---

## Gap 1: Player Availability

**The problem.** TimesFM has no concept of Cole Palmer, Enzo Fernandez, or Reece James. A sequence dip caused by injuries looks identical to a dip caused by a difficult fixture run. The model cannot distinguish them, and therefore cannot correctly forecast the recovery curve when a key player returns.

**The patch.** Add a simple `key_players_unavailable` integer column to the data CSV. This is a count of first-choice starters who are injured or suspended for that match. You update this manually (or scrape from a source like Transfermarkt) before each matchweek.

```python
# In the data CSV, add a column:
# key_players_unavailable: integer (0-4 is a reasonable range)
# 0 = full squad available
# 1-2 = minor disruption
# 3+ = significant disruption

# Then pass as a covariate to TimesFM:
point_forecast, quantile_forecast = model.forecast(
    horizon=7,
    inputs=[cumulative_points_array],
    # Future versions: covariates=[injury_count_array]
)
```

**Current status.** TimesFM 2.5 supports covariates in its API. The `run_real_forecast.py` script is structured to accept this extension — the `key_injuries` column already exists in the data schema from `data_loader.py`.

---

## Gap 2: Fixture Difficulty

**The problem.** The model treats MW7 (Chelsea 2-1 Liverpool) and MW12 (Chelsea 2-0 Burnley) as equivalent 3-point events. But the difficulty of earning those 3 points is vastly different, and the *expected* points against each opponent should inform the forecast differently.

**The patch.** Before each matchweek, append the opponent's current points total (or last-5 PPG) as a covariate column. This is a one-line lookup from the same FBref data source.

```python
# Opponent strength lookup (example)
opponent_pts = {
    'Arsenal': 70, 'Manchester City': 61, 'Tottenham Hotspur': 30,
    'Burnley': 20, 'Wolves': 17, ...
}
df['opponent_pts'] = df['opponent'].map(opponent_pts)
```

This gives the model a signal: "this 3-point win came against a 70-point opponent" vs. "this 3-point win came against a 17-point opponent."

---

## Gap 3: Home/Away Split

**The problem.** Chelsea's home record (6-5-4, 23 pts) and away record (7-4-5, 25 pts) are actually fairly balanced this season, but in general the home/away split is one of the most reliable structural features in football. The model currently treats all matches as equivalent.

**The patch.** Run two parallel forecast series — one on home-only cumulative points, one on away-only — and combine them. This is a 10-line code change.

```python
home_pts = df[df['venue'] == 'Home']['cumulative_points_home'].values
away_pts = df[df['venue'] == 'Away']['cumulative_points_away'].values

home_forecast, _ = model.forecast(horizon=4, inputs=[home_pts])
away_forecast, _ = model.forecast(horizon=3, inputs=[away_pts])
```

The remaining 7 fixtures (MW32-38) can be tagged H/A, and the combined forecast becomes more fixture-aware.

---

## Gap 4: Competition Fatigue

**The problem.** The `days_rest` column partially captures this, but it does not distinguish between "3 days rest after a routine league match" and "3 days rest after a 120-minute UCL knockout tie." The physical and psychological cost is different.

**The patch.** Add a `competition_type` column to the data and a `fatigue_weight` derived from it:

```python
fatigue_weights = {
    'Premier League': 1.0,
    'Champions League': 1.3,   # Higher physical cost
    'FA Cup': 0.9,
    'EFL Cup': 0.7,
}
df['fatigue_weight'] = df['competition'].map(fatigue_weights)
df['adjusted_days_rest'] = df['days_rest'] / df['fatigue_weight']
```

This adjusted rest metric becomes a more honest covariate.

---

## Gap 5: Tactical / Regime Shifts

**The problem.** When a manager changes formation or a key signing integrates, the team's underlying performance characteristics shift in ways that look like noise to a sequence model. The model will try to average through a regime change rather than recognize it.

**The patch.** This is the lowest-priority gap and the hardest to formalize. The simplest approach is a manual `regime_flag` column — a binary that resets to 1 when a significant tactical or personnel change occurs. You can then split the historical series at that point and only feed the model the post-regime data as context.

```python
# Only use post-regime data as context
regime_start_mw = 22  # e.g., after January transfer window
context_df = df[df['matchweek'] >= regime_start_mw]
```

---

## Gap 6: Opponent Form

**The problem.** A fixture against Everton in MW31 (Chelsea lost 0-3) looks like a routine match on paper but Everton were in strong form (WLWWLL, 46 pts). The model sees Chelsea's sequence dip but cannot attribute it to opponent strength.

**The patch.** Scrape the opponent's last-5 PPG from the same FBref source and append it as a column. This is the most tractable covariate to add next.

```python
# From the EPL table data already on the portfolio:
opponent_last5_ppg = {
    'Everton': 1.4, 'Arsenal': 2.2, 'Burnley': 0.6, ...
}
df['opponent_form_l5'] = df['opponent'].map(opponent_last5_ppg)
```

---

## Implementation Priority

The gaps above are ordered by implementation effort and expected impact:

| Priority | Gap | Effort | Expected forecast improvement |
|---|---|---|---|
| 1 | Fixture difficulty (opponent pts) | 30 min | High — changes the expected baseline per match |
| 2 | Home/away split | 1 hour | Medium — adds structural fixture awareness |
| 3 | Player availability | Ongoing | High when injuries are significant |
| 4 | Competition fatigue | 30 min | Medium — already partially captured |
| 5 | Opponent form | 30 min | Medium — refines difficulty signal |
| 6 | Regime/tactical shifts | Judgment call | Low-medium — hard to formalize |

The next logical step is to implement **Priority 1 and 2** in `run_real_forecast.py` and observe how the forecast distribution shifts compared to the baseline.

---

*Data: FBref · Chelsea 2025-26 PL · MW1-31 · Last updated: Apr 2026*
