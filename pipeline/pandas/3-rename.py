#!/usr/bin/env python3
"""
rename the data frame
"""

import pandas as pd


def rename(df):
    """
    inside the function
    """
    modified_df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
    return modified_df[["Datetime", "Close"]]
