#!/usr/bin/env python3
"""
slicing data frame
"""


def slice(df):
    """
    inside the func
    """
    df = df[['High', 'Low', 'Close', 'Volume_BTC']]
    df_sliced = df.iloc[::60]
    return df_sliced
