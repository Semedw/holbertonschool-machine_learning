#!/usr/bin/env python3
"""
from numpy
"""

import pandas as pd
import string

def from_numpy(array):
    """
    inside the function
    """
    labels = string.ascii_uppercase
    df = pd.DataFrame({labels : array})
    return df
