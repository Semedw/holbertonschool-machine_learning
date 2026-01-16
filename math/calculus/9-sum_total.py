#!/usr/bin/env python3
"""
total sum
"""


def summation_i_squared(n):
    """
    sum of squared numbers
    """
    if isinstance(n, int):
        s = n * (n ** 2 + 1) / 2
        return s
    return None
