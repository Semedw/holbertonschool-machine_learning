#!/usr/bin/env python3
"""
slicing data frame
"""


def slice(df):
    """
    inside the func
    """
    cols = ['High', 'Low', 'Close', 'Volume_(BTC)']
    df = df[cols]
    df_sliced = df.iloc[::60]
    return df_sliced
