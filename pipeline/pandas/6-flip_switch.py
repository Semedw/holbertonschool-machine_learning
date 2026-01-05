#!/usr/bin/env python3
"""
flip it and switch it
"""


def flip(df):
    """
    inside the func
    """
    df_sorted = df.sort_values(by="Timestamp", ascending=False)
    df_transposed = df_sorted.T
    return df_transposed
