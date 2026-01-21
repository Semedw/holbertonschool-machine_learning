#!/usr/bin/env python3
"""
the whole barn
"""


def same_shape(a, b):
    if isinstance(a, list) != isinstance(b, list):
        return False

    if not isinstance(a, list):
        return True

    if len(a) != len(b):
        return False

    for x, y in zip(a, b):
        if not same_shape(x, y):
            return False

    return True

def add_matrices(mat1, mat2):
    """
    adding matrices
    """
    if not same_shape(mat1, mat2):
        return None
    if not isinstance(mat1, list):
        return mat1 + mat2
    return [add_matrices(matrix1, matrix2) for matrix1, matrix2 in zip(mat1,
                                                                      mat2)]
