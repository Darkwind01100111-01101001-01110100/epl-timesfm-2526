import pandas as pd
import numpy as np

def generate_mock_chelsea_data(matchweeks=31):
    """
    Generates a mock dataset representing Chelsea's season up to a specific matchweek.
    Includes points, goals, and covariates like fixture congestion and injuries.
    """
    np.random.seed(42) # For reproducibility
    
    # Generate dates (approximate 1 match per week, some mid-week)
    start_date = pd.to_datetime('2025-08-15')
    dates = [start_date + pd.Timedelta(days=int(i*7 + np.random.normal(0, 1.5))) for i in range(matchweeks)]
    
    # Generate points (0, 1, 3) based on rough Chelsea form (13W, 9D, 9L)
    # This is a simplified distribution to match the 48 points in 31 games
    outcomes = ['W']*13 + ['D']*9 + ['L']*9
    np.random.shuffle(outcomes)
    
    points_map = {'W': 3, 'D': 1, 'L': 0}
    points = [points_map[o] for o in outcomes]
    cumulative_points = np.cumsum(points)
    
    # Generate goals for and against
    gf = [np.random.poisson(1.7) for _ in range(matchweeks)] # Avg ~53 goals in 31 games
    ga = [np.random.poisson(1.2) for _ in range(matchweeks)] # Avg ~38 goals in 31 games
    
    # Covariates
    # Days since last match (fixture congestion)
    days_rest = [7] # First match
    for i in range(1, matchweeks):
        days_rest.append((dates[i] - dates[i-1]).days)
        
    # Injuries (0 to 5 key players out)
    injuries = np.random.poisson(1.5, matchweeks)
    injuries = np.clip(injuries, 0, 5)
    
    # Create DataFrame
    df = pd.DataFrame({
        'matchweek': range(1, matchweeks + 1),
        'date': dates,
        'result': outcomes,
        'points_earned': points,
        'cumulative_points': cumulative_points,
        'goals_for': gf,
        'goals_against': ga,
        'days_rest': days_rest,
        'key_injuries': injuries
    })
    
    return df

if __name__ == "__main__":
    df = generate_mock_chelsea_data()
    df.to_csv('../data/chelsea_mock_data.csv', index=False)
    print(f"Generated mock data for {len(df)} matchweeks.")
