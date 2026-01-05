#!/usr/bin/python3
"""
slicing data frame
"""


def slice(df):
    """
    inside the func
    """
    df = df[["High", "Low", "Close", "Volume_BTC"]]
    df = df.iloc[60]
    return df
