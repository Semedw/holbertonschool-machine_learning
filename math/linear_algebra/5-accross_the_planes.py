#!/usr/bin/env python3
"""
adding 2d matrices
"""


def add_matrices(mat1, mat2):
    """
    adding matrices
    """
    if len(mat1) != len(mat2):
        return None
    result = []
    for i in range(len(mat1)):
        s = []
        for j in range(len(mat1[i])):
            s.append(mat1[i][j] + mat2[i][j])
        result.append(s)
    return result
print(add_matrices([[1, 2], [3, 4]], [[5, 6], [7, 8]]))
