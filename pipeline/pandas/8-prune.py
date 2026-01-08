#!/usr/bin/env python3
"""
filtering the dataset
"""


def prune(df):
    """
    removing(filtering) close entries with NaN values
    """
    df_removed = df.dropna(subset="Close")
    return df
