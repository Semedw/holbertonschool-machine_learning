#!/usr/bin/env python3
'''
calculating specificity
'''

import numpy as np


def specificity(confusion):
    '''
    inside the func
    '''
    res = []
    for c in range(len(confusion)):
        fp = sum(confusion[:, c]) - confusion[c, c]
        tn = sum(np.delete(sum(confusion)-confusion[c, :], c))
        res.append(tn / (tn + fp))
    return np.array(res)
