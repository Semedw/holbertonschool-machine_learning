#!/usr/bin/env python3
"""
flip it and switch it
"""

def flip(df):
    df = df.sort_values(by="Timestamp", ascending=False)
    df_transposed = df.T
    return df_transposed
