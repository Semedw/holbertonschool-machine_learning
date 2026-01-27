#!usr/bin/env python3
"""
finding determinant
"""


def determinant(matrix):
    """
    finding the determinant of matrix
    """
    if isinstance(matrix[0], list):
        if len(matrix) != len(matrix[0]):
            raise ValueError('matrix must be a square')
        else:
            if len(matrix) == 1:
                return matrix[0][0]
            if len(matrix) == 2:
                return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


            

    else:
        raise TypeError('matrix must be a list of lists')