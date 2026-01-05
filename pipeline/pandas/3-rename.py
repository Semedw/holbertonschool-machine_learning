#!/usr/bin/env python3
"""
rename the data frame
"""

import pandas as pd


def rename(df):
    """
    inside the function
    """
    df.rename(columns={"Timestamp": "Datetime"})
    df = df[["Datetime", "Close"]]
    return df
