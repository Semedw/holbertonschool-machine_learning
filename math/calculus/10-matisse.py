#!/usr/bin/env python3
"""
derivative of polynomial
"""


def poly_derivative(poly):
    """
    calculating the derivative
    """
    if len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    der = []
    for i in range(1, len(poly)):
        der.append(poly[i] * i)
    return der
