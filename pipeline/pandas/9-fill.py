#!/usr/bin/env python3
"""
filling nan values
"""


def fill(df):
    """
    removing weighted_price column and filling nan values
    """
    df = df.drop('Weighted_Price', axis=1, errors='ignore')
    df["Close"] = df["Close"].fillna(df["Close"].shift(1))
    for col in ['High', 'Low', 'Open']:
        df[col] = df['Close']
    df[["Volume_(BTC)", "Volume_(Currency)"]] = (
        df[["Volume_(BTC)", "Volume_(Currency)"]].fillna(0)
        )
    return df
