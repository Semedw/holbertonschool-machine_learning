#!/usr/bin/env python3
"""
from file
"""

import pandas as pd


def from_file(filename, delimeter):
    data = pd.read_csv(filename, delimeter=delimeter)
    df = pd.DataFrame(data)

    return df
