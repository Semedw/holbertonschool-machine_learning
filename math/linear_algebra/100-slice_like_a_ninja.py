#!/usr/bin/env python3
"""
slice like a ninja
"""


def np_slice(matrix, axes={}):
    """
    slicing matrix along specific axes
    """
    newMat = matrix
    result = []
    for key, value in axes.items(): 
        sec = slice(*value)
        if key == 0:
            newMat = newMat[sec]
        elif key == 1:
            newMat = newMat[:, sec]
            #result = matrix[:][sec]
        else:
            newMat = newMat[:, :, sec]
            #result = matrix[:][:][sec]
    return newMat
