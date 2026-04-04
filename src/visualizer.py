import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_points_trajectory(historical_df, forecast_df, save_path=None):
    """
    Plots the historical points trajectory alongside the TimesFM forecast
    with 10th-90th percentile confidence bands.
    """
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Extract data
    hist_mw = historical_df['matchweek']
    hist_pts = historical_df['cumulative_points']
    
    fcst_mw = forecast_df['matchweek']
    fcst_mean = forecast_df['forecast_mean']
    fcst_lower = forecast_df['forecast_lower_10']
    fcst_upper = forecast_df['forecast_upper_90']
    
    # Plot historical data
    plt.plot(hist_mw, hist_pts, marker='o', linestyle='-', color='#034694', label='Actual Points (Chelsea)', linewidth=2)
    
    # Connect the last historical point to the first forecast point
    last_hist_mw = hist_mw.iloc[-1]
    last_hist_pt = hist_pts.iloc[-1]
    
    conn_mw = [last_hist_mw, fcst_mw.iloc[0]]
    conn_pts = [last_hist_pt, fcst_mean.iloc[0]]
    plt.plot(conn_mw, conn_pts, linestyle='--', color='#FFA500')
    
    # Plot forecast mean
    plt.plot(fcst_mw, fcst_mean, marker='o', linestyle='--', color='#FFA500', label='TimesFM Forecast (Mean)', linewidth=2)
    
    # Plot confidence bands (10th to 90th percentile)
    plt.fill_between(fcst_mw, fcst_lower, fcst_upper, color='#FFA500', alpha=0.2, label='80% Confidence Interval')
    
    # Formatting
    plt.title('Chelsea FC: Cumulative Points Trajectory (TimesFM 2.5 Forecast)', fontsize=16, pad=15)
    plt.xlabel('Matchweek', fontsize=12)
    plt.ylabel('Cumulative Points', fontsize=12)
    
    # Add target lines (e.g., Europa League qualification ~60 pts)
    plt.axhline(y=60, color='gray', linestyle=':', alpha=0.7, label='Europa League Target (60 pts)')
    
    # Set axis limits and ticks
    plt.xlim(0, 39)
    plt.xticks(np.arange(0, 40, 2))
    plt.ylim(0, max(fcst_upper.max() + 5, 70))
    
    # Add annotations for key points
    plt.annotate(f'Current: {last_hist_pt} pts', 
                 xy=(last_hist_mw, last_hist_pt),
                 xytext=(last_hist_mw - 5, last_hist_pt + 5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
                 
    plt.annotate(f'Projected: {int(fcst_mean.iloc[-1])} pts', 
                 xy=(fcst_mw.iloc[-1], fcst_mean.iloc[-1]),
                 xytext=(fcst_mw.iloc[-1] - 6, fcst_mean.iloc[-1] - 10),
                 arrowprops=dict(facecolor='#FFA500', shrink=0.05, width=1, headwidth=5))
    
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    try:
        hist_df = pd.read_csv('../data/chelsea_mock_data.csv')
        fcst_df = pd.read_csv('../outputs/points_forecast.csv')
    except FileNotFoundError:
        print("Data files not found. Run data_loader.py and forecaster.py first.")
        exit(1)
        
    plot_points_trajectory(hist_df, fcst_df, save_path='../outputs/points_trajectory.png')
