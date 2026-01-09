#!/usr/bin/env python3
"""
doing multiple things but concatenating is more significant
"""

import pandas as pd

index = __import__('10-index').index

def concat(df1, df2):
    """
    concatenating dataframes
    """
    df1 = index(df1)
    df2 = index(df2)
    concatenated_df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
    return concatenated_df
