import numpy as np
import pandas as pd

def add_fourier_seasonality(df, period=52, order=2):
    """
    Adds Fourier seasonality terms.

    Parameters:
    - df: DataFrame with time index
    - period: seasonal period (52 for weekly)
    - order: number of sine/cosine pairs

    Returns:
    - DataFrame with seasonality columns
    """
    t = np.arange(len(df))

    for k in range(1, order + 1):
        df[f"sin_{period}_{k}"] = np.sin(2 * np.pi * k * t / period)
        df[f"cos_{period}_{k}"] = np.cos(2 * np.pi * k * t / period)

    return df
