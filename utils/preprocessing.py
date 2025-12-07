import pandas as pd

def smooth_and_clean_data(df):
    """
    Preprocess the forecast dataframe: interpolate, smooth, and clean.
    Expects a DataFrame with a 'time' column and irradiance columns (GHI, DNI, DHI).
    """
    df = df.copy()
    df.set_index('time', inplace=True)

    # Interpolate any missing values
    df.interpolate(method='linear', inplace=True)

    # Smooth data using a rolling mean over 3 intervals (1.5 hours for 30-min resolution)
    df = df.rolling(window=3, min_periods=1).mean()

    df = df.reset_index()
    return df
