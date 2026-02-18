#!/usr/bin/env python3
"""
finding sensitivity
"""

import numpy as np


def sensitivity(confusion):
    '''
    inside the func
    '''

    for c in range(len(confusion)):
        tp = confusion[c, c]
        fn = sum(confusion[c,:]) - confusion[c,c]
    return tp/(tp+fn)
