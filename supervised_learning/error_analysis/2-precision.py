#!/usr/bin/env python3
'''
calculating precision
'''

import numpy as np


def precision(confusion):
    '''
    inside the func
    '''
    res = []
    for c in range(len(confusion)):
        tp = confusion[c, c]
        fp = sum(confusion[:, c]) - confusion[c, c]
        res.append(tp / (tp + fp))
    return np.array(res)
