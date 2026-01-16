#!/usr/bin/env python3
"""
total sum
"""


def summation_i_squared(n):
    """
    sum of squared numbers
    """
    if isinstance(n, int):
        if n >= 1:
            s = n * (n + 1) * (2 * n + 1) / 6
            return s
        else:
            return None
    return None
