import pandas as pd
import numpy as np
import timesfm
from timesfm.timesfm_2p5 import timesfm_2p5_torch

class ChelseaForecaster:
    def __init__(self, model_path="google/timesfm-2.5-200m-pytorch"):
        """
        Initializes the TimesFM model for Chelsea forecasting.
        """
        print(f"Loading TimesFM model from {model_path}...")
        self.model = timesfm.timesfm_2p5.timesfm_2p5_torch.TimesFM_2p5_200M_torch.from_pretrained(
            model_path, torch_compile=True
        )
        
        # Configure the forecast
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=1024,
                max_horizon=256,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        print("Model compiled successfully.")

    def forecast_points(self, df, horizon=7):
        """
        Forecasts cumulative points for the remaining matchweeks.
        """
        # Ensure data is sorted chronologically
        df = df.sort_values('matchweek')
        
        # Extract the target variable (cumulative points)
        historical_points = df['cumulative_points'].values
        
        print(f"Forecasting {horizon} matchweeks ahead based on {len(historical_points)} historical data points...")
        
        # Run the forecast
        point_forecast, quantile_forecast = self.model.forecast(
            horizon=horizon,
            inputs=[historical_points]
        )
        
        # Extract results (shape is [batch_size, horizon] or [batch_size, horizon, num_quantiles])
        mean_forecast = point_forecast[0]
        
        # Quantiles: 10th to 90th
        # quantile_forecast[0, :, 0] is the 10th percentile (lower bound)
        # quantile_forecast[0, :, -1] is the 90th percentile (upper bound)
        lower_bound = quantile_forecast[0, :, 0]
        upper_bound = quantile_forecast[0, :, -1]
        
        # Create a DataFrame for the results
        last_matchweek = df['matchweek'].max()
        future_matchweeks = range(last_matchweek + 1, last_matchweek + horizon + 1)
        
        results_df = pd.DataFrame({
            'matchweek': future_matchweeks,
            'forecast_mean': mean_forecast,
            'forecast_lower_10': lower_bound,
            'forecast_upper_90': upper_bound
        })
        
        return results_df

if __name__ == "__main__":
    # Test the forecaster with mock data
    try:
        df = pd.read_csv('../data/chelsea_mock_data.csv')
    except FileNotFoundError:
        print("Mock data not found. Run data_loader.py first.")
        exit(1)
        
    forecaster = ChelseaForecaster()
    
    # Forecast the remaining 7 games of a 38-game season
    forecast_df = forecaster.forecast_points(df, horizon=7)
    
    print("\nForecast Results (Next 7 Matches):")
    print(forecast_df)
    
    forecast_df.to_csv('../outputs/points_forecast.csv', index=False)
    print("Saved forecast to outputs/points_forecast.csv")
