#!/usr/bin/env python3
'''
f1 score
'''

import numpy as np

def f1_score(confusion):
    '''
    inside the func
    '''
    res = []
    for c in range(len(confusion)):
        tp = confusion[c, c]
        fp = sum(confusion[:,c]) - confusion[c, c]
        fn = sum(confusion[c,:]) - confusion[c, c]
        tn = sum(np.delete(sum(confusion)-confusion[c, :],c))
        res.append(2*tp/(2*tp + fp + fn))
    return np.array(res)