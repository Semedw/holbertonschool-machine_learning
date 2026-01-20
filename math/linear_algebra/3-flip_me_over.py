#!/usr/bin/env python3
"""
transposing the matrix
"""


def matrix_transpose(matrix):
    """
    flipping over the diagonal
    """
    t_matrix = []
    for i in range(len(matrix[0])):
        row = []
        for j in range(len(matrix)):
            row.append(matrix[j][i])
        t_matrix.append(row)
    return t_matrix
