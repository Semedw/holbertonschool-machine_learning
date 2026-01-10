#!/usr/bin/env python3
"""
analyzing the dataframe
"""


def analyze(df):
    """
    analyzing all columns except timestamp
    """
    df = df[['Open', 'High', 'Low', 'Close', 'Volume_(BTC)',
             'Volume_(Currency)', 'Weighted_Price']]
    df = df.describe()
    return df
