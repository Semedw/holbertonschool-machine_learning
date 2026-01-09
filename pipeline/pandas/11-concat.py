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
    df1 = df1.set_index('Timestamp')
    df2 = df2.set_index('Timestamp')
    concatenated_df = pd.concat([df1, df2], keys=['coinbase', 'bitstamp'])
    return concatenated_df
