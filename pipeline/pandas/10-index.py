#!/usr/bin/env python3
"""
indexing the timestamp column
"""


def index(df):
    """
    set the timestamp column as the index of dataframe
    """
    df['Timestamp'] = df.index
    return df
