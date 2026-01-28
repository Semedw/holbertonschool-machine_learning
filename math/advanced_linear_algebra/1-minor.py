#!/usr/bin/env python3
"""
function for finding minor
"""


def determinant(matrix):
    """ 
    finding the determinant of matrix
    """
    if isinstance(matrix, list):
        for i in matrix:
            if not isinstance(i, list):
                raise TypeError('matrix must be a list of lists')
        k = len(matrix)
        for i in matrix:
            if len(i) == 0:
                return 1
            if len(i) != k:
                raise ValueError('matrix must be a square matrix')
        if len(matrix) == 1:
            return matrix[0][0]
        if len(matrix) == 2:
            det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
            return det 
        det = 0 
        for col in range(len(matrix)):
            sub = [row[:col] + row[col+1:] for row in matrix[1:]]
            det += ((-1)**col * matrix[0][col] * determinant(sub))
        return det 
    else:
        raise TypeError('matrix must be a list of lists')


def minor(matrix):
    """
    finding minor
    """
    if isinstance(matrix, list):
        for i in matrix:
            if not isinstance(i, list):
                raise TypeError('matrix must be a list of lists')
            if len(i) != len(matrix) or len(i) == 0:
                raise ValueError('matrix must be a non-empty square matrix')
        rows = len(matrix)
        cols = len(matrix[0])
        minor = []
        if len(matrix) == 1:
            return [[1]]
        for r in range(rows):
            new = []
            for c in range(cols):
                sub = [row[:c] + row[c+1:] for i, row in enumerate(matrix) if
                       i != r]
                if len(sub) == 1:
                    new.append(sub[0][0])
                else:
                    new.append(determinant(sub))
            minor.append(new)
        return minor
    else:
        raise TypeError('matrix must be a list of lists')
