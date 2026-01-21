#!/usr/bin/env python3
"""
the whole barn
"""


def add_matrices(mat1, mat2):
    """
    adding matrices
    """
    if len(matrix1) != len(matrix2):
        return None
    return [add_matrice(matrix1, matrix2) for matrix1, matrix2 in zip(mat1,
                                                                      mat2)]
