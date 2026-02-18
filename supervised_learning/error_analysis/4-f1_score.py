#!/usr/bin/env python3
'''
f1 score
'''

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision

def f1_score(confusion):
    '''
    inside the func
    '''
    pr = precision(confusion)
    rec = sensitivity(confusion)

    f_value = 2/(1/pr + 1/rec)
    return f_value