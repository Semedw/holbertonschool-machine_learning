#!/usr/bin/env python3
"""
ridin' bareback
"""


def mat_mul(mat1, mat2):
    """
    multiplying matrices
    """
    if len(mat1[0]) != len(mat2):
        return None
    result = []
    for i in range(len(mat1)):
        for j in range(len(mat1[i])):
            s= []
            for k in range(len(mat2[j])):
                s.append(mat1[i][j] * mat2[j][k])
            result.append(s)
    last = []
    for i in range(0, len(result)-1, len(mat2)):
        s = []
        for j in range(len(result[i])):
            s.append(result[i][j] + result[i+1][j])
        last.append(s)
    return last
