#!/usr/bin/env python3
"""
finding determinant
"""


def determinant(matrix):
    """
    finding the determinant of matrix
    """
    if isinstance(matrix, list):
        if isinstance(matrix[0], list):
            k = len(matrix)
            for i in matrix:
                if len(i) != k:
                    raise ValueError('matrix must be a square matrix')
            else:
                if len(matrix) == 1:
                    return matrix[0][0]
                if len(matrix) == 2:
                    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        else:
            raise TypeError('matrix must be a list of lists')
    else:
        raise TypeError('matrix must be a list of lists')