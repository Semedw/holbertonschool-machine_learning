#!/usr/bin/env python3
"""
getting cozy
"""


def cat_matrices2D(mat1, mat2, axis=0):
    if axis == 0:
        result = mat1 + mat2
    else:
        result = []
        if len(mat1) != len(mat2):
            return None
        for i in range(len(mat1)):
            result.append(mat1[i] + mat2[i])
    return result
