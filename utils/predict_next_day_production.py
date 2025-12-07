import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def clean_and_smooth_irradiance(df, ghi_column='GHI', time_column='time'):
    """
    Clean and smooth GHI values.
    - Set GHI to 0 before 6:00 and after 18:40.
    - Apply rolling mean smoothing.
    """
    df[time_column] = pd.to_datetime(df[time_column])
    df['hour'] = df[time_column].dt.hour + df[time_column].dt.minute / 60.0

    # Set GHI to 0 outside sunrise-sunset window
    df.loc[(df['hour'] < 6) | (df['hour'] > 18.66), ghi_column] = 0

    # Smooth GHI (rolling average)
    df[ghi_column] = df[ghi_column].rolling(window=3, center=True, min_periods=1).mean()

    return df

def predict_solar_production(df, model, scaler=None, ghi_column='GHI'):
    """
    Predict solar production based on GHI using the provided model.
    scaler: optional preprocessing (like StandardScaler or MinMaxScaler)
    """
    features = df[[ghi_column]].copy()
    
    # Optional scaling
    if scaler:
        features = scaler.transform(features)

    predictions = model.predict(features)

    df['Predicted Energy (Wh)'] = predictions

    return df

def plot_irradiance_and_production(df, time_column='time'):
    """
    Plot smoothed irradiance and predicted production on same chart.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(df[time_column], df['GHI'], label='Smoothed Irradiance (GHI)', color='skyblue')
    ax1.set_ylabel('GHI [W/m²]', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax1.set_xlabel('Time of Day')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    ax2 = ax1.twinx()
    ax2.plot(df[time_column], df['Predicted Energy (Wh)'], label='Predicted Energy', color='orange')
    ax2.set_ylabel('Predicted Energy [Wh]', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    plt.title("☀️ Smoothed Irradiance vs Predicted Solar Production (Next Day)")
    fig.tight_layout()
    plt.grid(True)
    plt.show()

# Example usage in Streamlit or a script
def run_prediction_pipeline(forecast_df, model, scaler=None):
    """
    Full pipeline to clean, smooth, predict and return forecast + predictions.
    `forecast_df` must have at least a 'time' and 'GHI' column (Global Horizontal Irradiance)
    """
    df = forecast_df.copy()
    df = clean_and_smooth_irradiance(df, ghi_column='GHI', time_column='time')
    df = predict_solar_production(df, model, scaler=scaler, ghi_column='GHI')
    return df
