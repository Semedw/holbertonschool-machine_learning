#!/usr/bin/env python3
"""
calculating integral
"""


def poly_integral(poly, C=0):
    """
    inside the func
    """
    if not isinstance(poly, list) or not isinstance(C, int):
        return None
    if len(poly) == 0:
        return None
    result = [0]
    for i in range(len(poly)):
        k = poly[i] / (i+1)
        if int(k) == k:
            result.append(int(k))
        else:
            result.append(k)
    return result
