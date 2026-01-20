#!/usr/bin/env python3
"""
cat's got your tongue
"""


def np_cat(mat1, mat2, axis=0):
    """
    concating matrices
    """
    result = np.concatenate((mat1, mat2), axis)
    return result
