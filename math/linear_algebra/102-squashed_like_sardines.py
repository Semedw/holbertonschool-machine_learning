#!/usr/bin/env python3
"""
squashed like sardines
"""


def same_shape(mat1, mat2):
    """'
    checks if the matrices are matching
    """
    if isinstance(mat1, list) != isinstance(mat2, list):
        return False

    if not isinstance(mat1, list):
       return True

    if len(mat1) != len(mat2):
        return False
    
    for x, y in zip(mat1, mat2):
        if not same_shape(x, y):
            return False

    return True

    
def cat_matrices(mat1, mat2, axis=0):
    """
    concatenating matrices
    """
    if not same_shape(mat1, mat2):
        return None
    return mat1 + mat2
