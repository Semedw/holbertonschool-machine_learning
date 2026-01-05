#!/usr/bin/env python3
"""
from numpy
"""

import pandas as pd


def from_numpy(array):
    """
    inside the function
    """
    df = pd.DataFrame(array)
    for i in range(len(df.columns)):
        df.columns = chr[65+i]
    return df
