#!/usr/bin/env python3
'''
Bayesian optimization
'''

import numpy as np

from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    '''
    Bayesian optimization class
    '''

    def __init__(self, f, X_init, y_init, bounds, ac_samples, l=1.0,
                 sigma_f=1.0, xsi=0.01, minimize=True):
        """
        init method for bayesian optimization
        Args:
            f: the black-box function to be optimized
            X_init: numpy.ndarray of shape (t, 1) representing the inputs
                    already sampled with the black-box function
            Y_init: numpy.ndarray of shape (t, 1) representing the outputs
                    of the black-box function for each input in X_init
                    t: the number of initial samples
            bounds: tuple of (min, max) representing the bounds of the space
                    in which to look for the optimal point
            ac_samples: the number of samples that should be analyzed during
                        acquisition
            l: the length parameter for the kernel
            sigma_f: the standard deviation given to the output of the
                    black-box function
            xsi: the exploration-exploitation factor for acquisition
            minimize: bool determining whether optimization should be performed
                    for minimization (True) or maximization (False)
        """
        # black-box function
        self.f = f

        # Gaussian Process
        self.gp = GP(X_init, y_init, l, sigma_f)

        # X_s all acquisition sample
        X_s = np.linspace(bounds[0], bounds[1], num=ac_samples)
        self.X_s = X_s.reshape(-1, 1)

        # exploration-explotation
        self.xsi = xsi

        # minimization versus maximization
        self.minimize = minimize

    def acquisition(self):
        """
        Public instance method def acquisition(self):
            that calculates the next best sample location:
            - Uses the Expected Improvement acquisition function

            - Returns: X_next, EI
                X_next is a numpy.ndarray of shape (1,) representing
                the next best sample point
                EI is a numpy.ndarray of shape (ac_samples,) containing
                the expected improvement of each potential sample
        """
        means, standard_deviations = self.gp.predict(self.X_s)
        if self.minimize:
            best_sample = min(self.gp.Y)
            improvement = best_sample - means - self.xsi
        else:
            best_sample = max(self.gp.Y)
            improvement = means - best_sample - self.xsi

        with np.errstate(divide='ignore'):
            Z = improvement / standard_deviations
            expected_improvement = (
                improvement * norm.cdf(Z) + standard_deviations * norm.pdf(Z)
            )
            expected_improvement[standard_deviations == 0.0] = 0.0

        X_next = self.X_s[np.argmax(expected_improvement)]

        return X_next, expected_improvement

    def optimize(self, iterations=100):
        """
        Public instance method def optimize(self, iterations=100):
            that performs Bayesian optimization for a given number of iterations:
            - Calls acquisition() to find the next best sample point
            - Updates the Gaussian Process with the new sample point and its output
            - Returns the optimal point and its expected improvement
        """
        X_all_s = []
        for i in range(iterations):
            # Find the next sampling point xt by optimizing the acquisition
            # function over the GP: xt = argmaxx μ(x | D1:t−1)

            x_opt, _ = self.acquisition()
            # If the next proposed point is one that has already been sampled,
            # optimization should be stopped early
            if x_opt in X_all_s:
                break

            y_opt = self.f(x_opt)

            # Add the sample to previous samples
            # D1: t = {D1: t−1, (xt, yt)} and update the GP
            self.gp.update(x_opt, y_opt)
            X_all_s.append(x_opt)

        if self.minimize is True:
            index = np.argmin(self.gp.Y)
        else:
            index = np.argmax(self.gp.Y)

        self.gp.X = self.gp.X[:-1]

        x_opt = self.gp.X[index]
        y_opt = self.gp.Y[index]

        return x_opt, y_opt
