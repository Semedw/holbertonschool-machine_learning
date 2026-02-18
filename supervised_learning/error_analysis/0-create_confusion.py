#!/usr/bin/env python3
"""
create confusion matrix
"""


import numpy as np


def create_confusion_matrix(labels, logits):
    '''
    inside the func
    '''
    return np.matmul(labels.T, logits)
