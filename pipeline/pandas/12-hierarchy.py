#!/usr/bin/env python3
"""
building hierarchy
"""

import pandas as pd

index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    rearranging multiindex
    """
    df1 = index(df1)
    df2 = index(df2)
    df1 = df1.loc[1417411980:1417417980]
    df2 = df2.loc[1417411980:1417417980]
    concatenated_df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'],
                                names=['exchange'])
   concatenated_df = concatenated_df.reorder_levels(order=['Timestamp',
   'exchange'])
    concatenated_df = concatenated_df.sort_index()
    return concatenated_df
