#!/usr/bin/env python3
'''
Bayesian optimization
'''

import numpy as np

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    '''
    Bayesian optimization class
    '''

    def __init__(self, f, X_init, y_init, bounds, ac_samples, l=1.0,
                 sigma_f=1.0, xsi=0.01, minimizer=True):
        self.f = f
        self.gp = GP(X_init, y_init, bounds, l, sigma_f, minimizer)
        self.ac_samples = ac_samples
        self.xsi = xsi
        self.minimizer = minimizer
