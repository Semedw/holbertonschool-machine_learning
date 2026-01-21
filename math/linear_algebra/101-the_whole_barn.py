#!/usr/bin/env python3
"""
the whole barn
"""


def add_matrices(mat1, mat2):
    """
    adding matrices
    """
    if len(mat1) != len(mat1):
        return None
    if not isinstance(mat1, list):
        return [a+b for a, b in zip(mat1, mat2)]
    return [add_matrice(matrix1, matrix2) for matrix1, matrix2 in zip(mat1,
                                                                      mat2)]
