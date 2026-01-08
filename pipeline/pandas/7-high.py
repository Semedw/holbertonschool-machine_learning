#!/usr/bin/env python3
"""
sort high
"""


def high(df):
    """
    sorting the dataframe by high values
    """
    sorted_df = df.sort_values(by="High", ascending=False)
    return sorted_df
