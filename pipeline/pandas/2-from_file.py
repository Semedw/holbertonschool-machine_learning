#!/usr/bin/env python3
"""
from file
"""

import pandas as pd


def from_file(filename, delimiter):
    data = pd.read_csv(filename, delimiter=delimiter)
    df = pd.DataFrame(data)

    return df
