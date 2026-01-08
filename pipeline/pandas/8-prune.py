#!/usr/bin/env python3
"""
filtering the dataset
"""


def prune(df):
    """
    removing(filtering) close entries with NaN values
    """
    df_filtered = df[df["Close"].notna()]
    return df_filtered
