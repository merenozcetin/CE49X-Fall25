"""
Lab 3: ERA5 Weather Data Analysis

This script analyzes ERA5 wind data for Berlin and Munich, including:
- Data loading and exploration
- Temporal aggregations (monthly, seasonal)
- Statistical analysis (extreme weather, diurnal patterns)
- Visualizations (time series, seasonal comparisons, wind direction analysis)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Define seasons mapping
SEASONS = {
    1: 'Winter', 2: 'Winter', 3: 'Spring',
    4: 'Spring', 5: 'Spring', 6: 'Summer',
    7: 'Summer', 8: 'Summer', 9: 'Fall',
    10: 'Fall', 11: 'Fall', 12: 'Winter'
}

def load_era5_data(berlin_path, munich_path):
    """
    Load ERA5 wind data for Berlin and Munich.
    
    Parameters:
    -----------
    berlin_path : str or Path
        Path to Berlin ERA5 data file
    munich_path : str or Path
        Path to Munich ERA5 data file
    
    Returns:
    --------
    tuple
        (berlin_df, munich_df) - DataFrames containing the loaded data
    """
    try:
        print("Loading ERA5 data files...")
        berlin_df = pd.read_csv(berlin_path)
        munich_df = pd.read_csv(munich_path)
        
        # Convert timestamp to datetime
        berlin_df['timestamp'] = pd.to_datetime(berlin_df['timestamp'])
        munich_df['timestamp'] = pd.to_datetime(munich_df['timestamp'])
        
        # Set timestamp as index for easier time-based operations
        berlin_df.set_index('timestamp', inplace=True)
        munich_df.set_index('timestamp', inplace=True)
        
        print(f"✓ Berlin data loaded: {len(berlin_df)} records")
        print(f"✓ Munich data loaded: {len(munich_df)} records")
        
        return berlin_df, munich_df
    except FileNotFoundError as e:
        print(f"Error: Data file not found: {e}")
        raise
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def calculate_wind_speed(u10m, v10m):
    """
    Calculate wind speed from u and v components.
    
    Wind speed = sqrt(u^2 + v^2)
    
    Parameters:
    -----------
    u10m : array-like
        U component of wind (m/s) - eastward wind
    v10m : array-like
        V component of wind (m/s) - northward wind
    
    Returns:
    --------
    array-like
        Wind speed (m/s)
    """
    return np.sqrt(u10m**2 + v10m**2)

def calculate_wind_direction(u10m, v10m):
    """
    Calculate wind direction from u and v components.
    
    Wind direction is measured in degrees, where:
    - 0° = North
    - 90° = East
    - 180° = South
    - 270° = West
    
    Parameters:
    -----------
    u10m : array-like
        U component of wind (m/s) - eastward wind
    v10m : array-like
        V component of wind (m/s) - northward wind
    
    Returns:
    --------
    array-like
        Wind direction in degrees (0-360)
    """
    # Calculate angle in radians, then convert to degrees
    # atan2 gives angle from positive x-axis (east), we want from north
    direction_rad = np.arctan2(u10m, v10m)
    direction_deg = np.degrees(direction_rad)
    # Convert to 0-360 range
    direction_deg = (direction_deg + 360) % 360
    return direction_deg

def explore_dataset(df, city_name):
    """
    Display basic information about the dataset.
    
    Parameters:
    -----------
    df : DataFrame
        Dataset to explore
    city_name : str
        Name of the city for display purposes
    """
    print(f"\n{'='*70}")
    print(f"Dataset Exploration: {city_name}")
    print(f"{'='*70}")
    print(f"Shape: {df.shape} (rows, columns)")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nDate range: {df.index.min()} to {df.index.max()}")
    print(f"\nSummary statistics:\n{df.describe()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nLast few rows:\n{df.tail()}")

def add_wind_metrics(df, city_name):
    """
    Add calculated wind speed and direction to the dataframe.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame with u10m and v10m columns
    city_name : str
        Name of the city (for display)
    
    Returns:
    --------
    DataFrame
        DataFrame with added wind_speed and wind_direction columns
    """
    df['wind_speed'] = calculate_wind_speed(df['u10m'], df['v10m'])
    df['wind_direction'] = calculate_wind_direction(df['u10m'], df['v10m'])
    print(f"✓ Added wind_speed and wind_direction for {city_name}")
    return df

def calculate_monthly_averages(df, city_name):
    """
    Calculate monthly averages for wind speed.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame with wind_speed column and datetime index
    city_name : str
        Name of the city
    
    Returns:
    --------
    DataFrame
        DataFrame with monthly averages
    """
    monthly_avg = df.groupby(df.index.to_period('M'))['wind_speed'].mean()
    monthly_avg.index = monthly_avg.index.to_timestamp()
    print(f"\nMonthly average wind speeds for {city_name}:")
    print(monthly_avg)
    return monthly_avg

def calculate_seasonal_averages(df, city_name):
    """
    Calculate seasonal averages for wind speed.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame with wind_speed column and datetime index
    city_name : str
        Name of the city
    
    Returns:
    --------
    DataFrame
        DataFrame with seasonal averages
    """
    # Add season column
    df['season'] = df.index.month.map(SEASONS)
    seasonal_avg = df.groupby('season')['wind_speed'].mean()
    
    # Order seasons properly
    season_order = ['Spring', 'Summer', 'Fall', 'Winter']
    seasonal_avg = seasonal_avg.reindex([s for s in season_order if s in seasonal_avg.index])
    
    print(f"\nSeasonal average wind speeds for {city_name}:")
    print(seasonal_avg)
    return seasonal_avg

def identify_extreme_weather(df, city_name, top_n=10):
    """
    Identify days/periods with extreme weather conditions (highest wind speeds).
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame with wind_speed column and datetime index
    city_name : str
        Name of the city
    top_n : int
        Number of top extreme events to display
    
    Returns:
    --------
    DataFrame
        DataFrame with top extreme weather events
    """
    # Find highest wind speeds
    df_sorted = df.nlargest(top_n, 'wind_speed')[['wind_speed', 'u10m', 'v10m', 'wind_direction']]
    
    print(f"\n{'='*70}")
    print(f"Top {top_n} Extreme Wind Events for {city_name}")
    print(f"{'='*70}")
    print(df_sorted)
    
    # Calculate daily maximum wind speeds
    daily_max = df.groupby(df.index.date)['wind_speed'].max()
    top_daily_max = daily_max.nlargest(top_n)
    
    print(f"\nTop {top_n} Daily Maximum Wind Speeds for {city_name}:")
    for date, speed in top_daily_max.items():
        print(f"  {date}: {speed:.2f} m/s")
    
    return df_sorted

def calculate_diurnal_patterns(df, city_name):
    """
    Calculate diurnal (daily) patterns in wind speed.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame with wind_speed column and datetime index
    city_name : str
        Name of the city
    
    Returns:
    --------
    Series
        Series with hourly average wind speeds (0-23)
    """
    # Extract hour from timestamp
    df['hour'] = df.index.hour
    diurnal_avg = df.groupby('hour')['wind_speed'].mean()
    
    print(f"\nDiurnal pattern (hourly averages) for {city_name}:")
    for hour, speed in diurnal_avg.items():
        print(f"  Hour {hour:02d}:00 - {speed:.2f} m/s")
    
    return diurnal_avg

def visualize_time_series_monthly(berlin_monthly, munich_monthly):
    """
    Create time series plot of monthly average wind speeds for both cities.
    
    Parameters:
    -----------
    berlin_monthly : Series
        Monthly averages for Berlin
    munich_monthly : Series
        Monthly averages for Munich
    """
    plt.figure(figsize=(14, 6))
    plt.plot(berlin_monthly.index, berlin_monthly.values, 
             marker='o', linewidth=2, label='Berlin', color='#1f77b4')
    plt.plot(munich_monthly.index, munich_monthly.values, 
             marker='s', linewidth=2, label='Munich', color='#ff7f0e')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Average Wind Speed (m/s)', fontsize=12)
    plt.title('Monthly Average Wind Speeds: Berlin vs Munich', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('monthly_wind_speeds.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved visualization: monthly_wind_speeds.png")
    plt.show()

def visualize_seasonal_comparison(berlin_seasonal, munich_seasonal):
    """
    Create seasonal comparison bar chart.
    
    Parameters:
    -----------
    berlin_seasonal : Series
        Seasonal averages for Berlin
    munich_seasonal : Series
        Seasonal averages for Munich
    """
    # Prepare data for plotting
    seasons = berlin_seasonal.index
    x = np.arange(len(seasons))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width/2, berlin_seasonal.values, width, 
                    label='Berlin', color='#1f77b4', alpha=0.8)
    bars2 = plt.bar(x + width/2, munich_seasonal.values, width, 
                    label='Munich', color='#ff7f0e', alpha=0.8)
    
    plt.xlabel('Season', fontsize=12)
    plt.ylabel('Average Wind Speed (m/s)', fontsize=12)
    plt.title('Seasonal Average Wind Speeds: Berlin vs Munich', fontsize=14, fontweight='bold')
    plt.xticks(x, seasons)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('seasonal_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization: seasonal_comparison.png")
    plt.show()

def visualize_wind_direction(berlin_df, munich_df):
    """
    Create wind direction analysis visualization (wind rose-like plot).
    
    Parameters:
    -----------
    berlin_df : DataFrame
        Berlin data with wind_direction column
    munich_df : DataFrame
        Munich data with wind_direction column
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), subplot_kw=dict(projection='polar'))
    
    cities = [berlin_df, munich_df]
    city_names = ['Berlin', 'Munich']
    colors = ['#1f77b4', '#ff7f0e']
    
    for ax, df, city_name, color in zip(axes, cities, city_names, colors):
        # Create wind direction bins (16 directions)
        bins = np.linspace(0, 360, 17)
        direction_counts, _ = np.histogram(df['wind_direction'], bins=bins)
        
        # Convert to radians and adjust for polar plot
        theta = np.deg2rad(bins[:-1])
        # Shift by 90 degrees so 0° (North) is at top
        theta = theta - np.pi/2
        
        # Normalize by total count for percentage
        direction_pct = (direction_counts / len(df)) * 100
        
        # Create bar plot
        bars = ax.bar(theta, direction_pct, width=np.deg2rad(360/16), 
                     color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Set labels
        ax.set_theta_zero_location('N')  # North at top
        ax.set_theta_direction(-1)  # Clockwise
        ax.set_thetagrids(np.arange(0, 360, 45), ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
        ax.set_title(f'Wind Direction Distribution - {city_name}', 
                    fontsize=12, fontweight='bold', pad=20)
        ax.set_ylabel('Frequency (%)', labelpad=30)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('wind_direction_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization: wind_direction_analysis.png")
    plt.show()

def visualize_diurnal_patterns(berlin_diurnal, munich_diurnal):
    """
    Create visualization of diurnal patterns for both cities.
    
    Parameters:
    -----------
    berlin_diurnal : Series
        Hourly averages for Berlin
    munich_diurnal : Series
        Hourly averages for Munich
    """
    plt.figure(figsize=(12, 6))
    plt.plot(berlin_diurnal.index, berlin_diurnal.values, 
             marker='o', linewidth=2, label='Berlin', color='#1f77b4')
    plt.plot(munich_diurnal.index, munich_diurnal.values, 
             marker='s', linewidth=2, label='Munich', color='#ff7f0e')
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Average Wind Speed (m/s)', fontsize=12)
    plt.title('Diurnal Pattern: Average Wind Speed by Hour of Day', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 24, 2))
    plt.xlim(-0.5, 23.5)
    plt.tight_layout()
    plt.savefig('diurnal_patterns.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization: diurnal_patterns.png")
    plt.show()

def compare_cities_statistics(berlin_df, munich_df):
    """
    Compare overall statistics between Berlin and Munich.
    
    Parameters:
    -----------
    berlin_df : DataFrame
        Berlin data with wind_speed column
    munich_df : DataFrame
        Munich data with wind_speed column
    """
    print(f"\n{'='*70}")
    print("Statistical Comparison: Berlin vs Munich")
    print(f"{'='*70}")
    
    stats = pd.DataFrame({
        'Berlin': [
            berlin_df['wind_speed'].mean(),
            berlin_df['wind_speed'].median(),
            berlin_df['wind_speed'].std(),
            berlin_df['wind_speed'].min(),
            berlin_df['wind_speed'].max(),
        ],
        'Munich': [
            munich_df['wind_speed'].mean(),
            munich_df['wind_speed'].median(),
            munich_df['wind_speed'].std(),
            munich_df['wind_speed'].min(),
            munich_df['wind_speed'].max(),
        ]
    }, index=['Mean', 'Median', 'Std Dev', 'Min', 'Max'])
    
    print(stats.round(2))
    print(f"\nBerlin has {'higher' if berlin_df['wind_speed'].mean() > munich_df['wind_speed'].mean() else 'lower'} average wind speed than Munich")
    print(f"Difference: {abs(berlin_df['wind_speed'].mean() - munich_df['wind_speed'].mean()):.2f} m/s")

def skyrim_repository_info():
    """
    Information about the Skyrim repository for weather forecasting.
    """
    info = """
    ======================================================================
    Skyrim Repository Information
    ======================================================================
    
    Skyrim (https://github.com/secondlaw-ai/skyrim) is a unified interface
    for running state-of-the-art large weather models like Graphcast, Pangu,
    and Fourcastnet. It provides access to initial conditions from sources
    like NOAA GFS and ECMWF IFS, allowing users to generate weather forecasts
    for different time horizons on consumer-grade hardware.
    
    This tool democratizes access to advanced weather forecasting capabilities
    that were previously only available on supercomputers, making it valuable
    for civil and environmental engineering projects that require weather
    predictions for planning, risk assessment, and resource management.
    
    ======================================================================
    """
    print(info)

def main():
    """
    Main function to run the ERA5 data analysis.
    """
    print("="*70)
    print("Lab 3: ERA5 Weather Data Analysis")
    print("="*70)
    
    # Define file paths (relative to script location)
    base_path = Path(__file__).parent.parent.parent
    berlin_path = base_path / "datasets" / "berlin_era5_wind_20241231_20241231.csv"
    munich_path = base_path / "datasets" / "munich_era5_wind_20241231_20241231.csv"
    
    # Load data
    try:
        berlin_df, munich_df = load_era5_data(berlin_path, munich_path)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return
    
    # Explore datasets
    explore_dataset(berlin_df, "Berlin")
    explore_dataset(munich_df, "Munich")
    
    # Add wind metrics
    berlin_df = add_wind_metrics(berlin_df, "Berlin")
    munich_df = add_wind_metrics(munich_df, "Munich")
    
    # Calculate temporal aggregations
    print("\n" + "="*70)
    print("TEMPORAL AGGREGATIONS")
    print("="*70)
    
    berlin_monthly = calculate_monthly_averages(berlin_df, "Berlin")
    munich_monthly = calculate_monthly_averages(munich_df, "Munich")
    
    berlin_seasonal = calculate_seasonal_averages(berlin_df, "Berlin")
    munich_seasonal = calculate_seasonal_averages(munich_df, "Munich")
    
    # Statistical analysis
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70)
    
    identify_extreme_weather(berlin_df, "Berlin")
    identify_extreme_weather(munich_df, "Munich")
    
    berlin_diurnal = calculate_diurnal_patterns(berlin_df, "Berlin")
    munich_diurnal = calculate_diurnal_patterns(munich_df, "Munich")
    
    compare_cities_statistics(berlin_df, munich_df)
    
    # Visualizations
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    visualize_time_series_monthly(berlin_monthly, munich_monthly)
    visualize_seasonal_comparison(berlin_seasonal, munich_seasonal)
    visualize_wind_direction(berlin_df, munich_df)
    visualize_diurnal_patterns(berlin_diurnal, munich_diurnal)
    
    # Skyrim repository information
    skyrim_repository_info()
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print("\nAll visualizations have been saved to the current directory.")
    print("Please check the generated PNG files.")

if __name__ == "__main__":
    main()

