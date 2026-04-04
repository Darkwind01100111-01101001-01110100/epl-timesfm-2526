# Chelsea FC Predictive Analytics with TimesFM 2.5

This repository demonstrates the integration of Google Research's **TimesFM 2.5** (Time Series Foundation Model) into a sports analytics workflow, specifically forecasting Chelsea FC's performance trajectory.

Instead of relying on simple linear projections or training custom deep learning models from scratch on limited data, this project uses a 200M parameter foundation model pre-trained on 100 billion real-world time points to generate zero-shot, uncertainty-aware forecasts.

## Project Structure

```
chelsea-timesfm/
├── data/
│   └── chelsea_mock_data.csv       # Simulated matchweek data (points, goals, covariates)
├── src/
│   ├── data_loader.py              # Script to generate/load the dataset
│   ├── forecaster.py               # TimesFM wrapper and prediction logic
│   └── visualizer.py               # Plotting trajectory and confidence bands
├── outputs/
│   ├── points_forecast.csv         # Raw output from TimesFM
│   └── points_trajectory.png       # Visualized forecast with 80% confidence interval
├── TIMESFM_CAPABILITIES.md         # Reference guide on TimesFM's strengths
└── README.md                       # This file
```

## Why TimesFM for Football Analytics?

1. **Zero-Shot Forecasting:** Predicts future points trajectories without needing to be trained on historical Premier League data.
2. **Quantile Forecasts:** Outputs confidence intervals (e.g., 10th to 90th percentile), moving beyond single-point predictions to show best/worst-case scenarios.
3. **Small Data Performance:** Excels in environments with limited data points (like a 38-game season) where traditional ML models would overfit.

## Setup & Installation

### Prerequisites
* Python 3.9+
* Git

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/chelsea-timesfm.git
   cd chelsea-timesfm
   ```

2. **Install core dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn torch
   ```

3. **Install TimesFM 2.5:**
   Currently, the 2.5 model requires installation directly from the Google Research GitHub repository:
   ```bash
   git clone https://github.com/google-research/timesfm.git
   cd timesfm
   pip install -e .
   ```

## Usage

The workflow is broken down into three steps:

1. **Generate Data:**
   ```bash
   cd src
   python data_loader.py
   ```
   *Creates `data/chelsea_mock_data.csv` representing 31 matchweeks of data.*

2. **Run Forecast:**
   ```bash
   python forecaster.py
   ```
   *Downloads the TimesFM 2.5 model (if not cached), processes the historical points, and outputs predictions for the remaining 7 matches to `outputs/points_forecast.csv`.*

3. **Visualize Results:**
   ```bash
   python visualizer.py
   ```
   *Generates `outputs/points_trajectory.png` showing the historical trend, the predicted mean, and the 80% confidence band.*

## Analytical Note

This repository is designed as an **exploratory sandbox**. The data provided is mock data structured to resemble Chelsea's 2025-26 season up to Matchweek 31. The architecture allows for easy swapping with live API data (e.g., via `rvest` or `BeautifulSoup`) to run real-time predictions as the season progresses.
