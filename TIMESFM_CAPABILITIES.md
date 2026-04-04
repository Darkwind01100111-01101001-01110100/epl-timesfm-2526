# TimesFM 2.5: The Foundation Model for Time-Series

This document serves as a quick reference guide to **TimesFM (Time Series Foundation Model)** by Google Research. It breaks down the core capabilities, what it excels at, and how it applies specifically to the Chelsea FC predictive analytics project.

## What is TimesFM?

TimesFM is a decoder-only foundation model pre-trained on 100 billion real-world time points. It is designed to understand the "language of time" and provide zero-shot forecasting — meaning it can predict future values for datasets it has never seen before, without requiring custom training.

The current version, **TimesFM 2.5**, is highly optimized:
*   **Size:** 200 million parameters (down from 500M in previous versions).
*   **Efficiency:** Fused QKV matrices for speed optimization; runs efficiently on CPU (~1.5 GB RAM) and GPU (~1 GB VRAM).
*   **Context:** Supports up to 16,384 context length (the historical data it looks back at).
*   **Architecture:** PyTorch-based, utilizing a continuous quantile head for uncertainty estimation.

## Core Capabilities: What is it great at?

TimesFM is particularly powerful for scenarios where traditional statistical models (like ARIMA or Prophet) struggle or where deep learning models would require massive, specialized datasets to train.

### 1. Zero-Shot Forecasting
You do not need to train or fine-tune TimesFM on your specific dataset. You pass in historical data (e.g., Chelsea's points trajectory), and it immediately outputs a forecast based on the generalized patterns it learned from 100 billion data points.

### 2. Quantile Forecasting (Uncertainty Bounds)
Instead of just giving a single "point forecast" (e.g., "Chelsea will finish with 65 points"), TimesFM provides a **quantile forecast**. It generates a range of possibilities (from the 10th to the 90th percentile), allowing you to visualize confidence intervals and best/worst-case scenarios.

### 3. Handling Multivariate Scenarios (Covariates)
TimesFM 2.5 supports incorporating covariates (external variables that influence the outcome). For example, when predicting points, you can include variables like:
*   Fixture congestion (days since last match)
*   Injury count
*   Opponent strength rating

### 4. Working with Limited Data
Traditional deep learning models require thousands of data points to learn patterns. TimesFM works excellently with small datasets (like a 38-game Premier League season), leveraging its pre-trained knowledge to infer trajectories from limited context.

## Application to Chelsea FC Analytics

For the Chelsea project, TimesFM shifts the focus from descriptive analytics ("what happened") to predictive and prescriptive analytics ("what is likely to happen, and what is the range of uncertainty").

### Example Use Cases

*   **Trajectory Projection:** Forecasting the end-of-season points total with a 90% confidence interval, updating dynamically after each matchweek.
*   **Fixture Congestion Impact:** Using covariates to model how 3 matches in 8 days affects expected points per game (PPG).
*   **Squad Health Modeling:** Predicting performance degradation based on the number of key injuries, identifying the critical threshold where form collapses.

## Code Implementation Pattern

The standard workflow for implementing TimesFM in Python involves three steps:

1.  **Initialize the Model:** Load the pre-trained weights from Hugging Face.
2.  **Configure the Forecast:** Set the context length, horizon (how far ahead to predict), and enable the quantile head.
3.  **Execute the Forecast:** Pass the historical arrays (and covariates) to the model and extract the point and quantile forecasts.

```python
import timesfm
import numpy as np

# 1. Initialize
tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)

# 2. Configure
tfm.compile(timesfm.ForecastConfig(
    max_context=512,
    max_horizon=10,
    normalize_inputs=True,
    use_continuous_quantile_head=True
))

# 3. Forecast
point_forecast, quantile_forecast = tfm.forecast(
    horizon=10,
    inputs=[historical_data_array]
)
```

## Summary

TimesFM 2.5 provides state-of-the-art predictive capabilities out of the box. By integrating it into the Chelsea analytics workflow, the project evolves from static dashboards to dynamic, uncertainty-aware forecasting systems.
