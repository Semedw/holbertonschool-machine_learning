#!/usr/bin/env python3
"""
function for finding minor
"""


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
        for r in range(rows):
            new = []
            for c in range(cols):
                sub = [row[:c] + row[c+1:] for i, row in enumerate(matrix) if
                       i != r]
                if len(sub) == 1:
                    new.append(sub[0][0])

    else:
        raise TypeError('matrix must be a list of lists')
