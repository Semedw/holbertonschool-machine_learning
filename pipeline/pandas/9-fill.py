#!/usr/bin/env python3
"""
filling nan values
"""


def fill(df):
    """
    removing weighted_price column and filling nan values
    """
    df.drop('Weighted_Price', axis=1)
    df["Close"] = df["Close"].fillna(df["Close"].shift(1))
    df[["High", "Low", "Open"]] = df["Close"]
    df[["Volume_(BTC)", "Volume_(Currency)"]] = df[["Volume_(BTC)", "Volume_(Currency)"]].fillna(0)
    return df
