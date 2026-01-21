#!/usr/bin/env python3
"""
squashed like sardines
"""


def cat_matrices(mat1, mat2, axis=0):
    """
    concating matrices
    """

    # Helper: get shape of a nested list
    def get_shape(mat):
        """
        getting the shape of matrice
        """
        if isinstance(mat, list):
            return [len(mat)] + get_shape(mat[0])
        return []

    shape1 = get_shape(mat1)
    shape2 = get_shape(mat2)

    # Check if axis is valid
    if axis < 0 or axis >= len(shape1):
        return None

    # Check compatibility on all axes except the concatenation axis
    for i, (s1, s2) in enumerate(zip(shape1, shape2)):
        if i != axis and s1 != s2:
            return None

    # Recursive concatenation
    def concat_recursive(m1, m2, ax):
        """
        concating the matrices
        """
        if ax == 0:
            return m1 + m2
        else:
            return [concat_recursive(m1[i], m2[i], ax-1)
                    for i in range(len(m1))]

    return concat_recursive(mat1, mat2, axis)
