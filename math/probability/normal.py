#!/usr/bin/env python3
"""
initializing normal dist class
"""


pi = 3.1415926536
e = 2.7182818285


class Normal:
    """
    inside the normal class
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        initializing the object
        """
        self.data = data
        self.mean = mean
        self.stddev = stddev

        if data is None:
            self.mean = mean
            self.stddev = stddev
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = sum(data) / len(data)
            sq = 0
            for value in data:
                sq += (value - self.mean)**2
            self.stddev = (sq/len(data))**(1/2)

    def z_score(self, x):
        """
        calculating z_score
        """
        z = (x - self.mean)/self.stddev
        return z

    def x_value(self, z):
        """
        calculatin x-value
        """
        x = z*self.stddev + self.mean
        return x

    def pdf(self, x):
        """
        calculating pdf
        """
        f_x = (1/(((2*pi)**0.5)*self.stddev))*e**(-0.5*((self.z_score(x))**2))
        return f_x
    
    def cdf(self, x):
        """
        calculating the cdf
        """
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))

        # Abramowitz and Stegun approximation
        t = 1 / (1 + 0.3275911 * abs(z))
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429

        erf = 1 - (
            (a1 * t +
             a2 * t**2 +
             a3 * t**3 +
             a4 * t**4 +
             a5 * t**5) *
            e ** (-z**2)
        )

        if z < 0:
            erf = -erf

        return 0.5 * (1 + erf)
