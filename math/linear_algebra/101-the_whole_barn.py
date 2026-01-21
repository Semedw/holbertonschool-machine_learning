#!/usr/bin/env python3
"""
the whole barn
"""


def add_matrices(mat1, mat2):
    """
    adding matrices
    """
    if isinstance(mat1, list) != isinstance(mat2, list):
        return None
    if not isinstance(mat1, list):
        return mat1 + mat2
    if len(mat1) != len(mat2):
        return None
    return [add_matrices(matrix1, matrix2) for matrix1, matrix2 in zip(mat1,
                                                                      mat2)]
