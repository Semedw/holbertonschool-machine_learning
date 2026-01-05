#!/usr/bin/env python3
"""
slicing data frame
"""

if __init__ == "__main__":

    def slice(df):
        """
        inside the func
         """
        cols = ['High', 'Low', 'Close', 'Volume_BTC']
        df = df[cols]
        df_sliced = df.iloc[::60]
        return df_sliced
