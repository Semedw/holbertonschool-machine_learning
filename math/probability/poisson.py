#!/usr/bin/env python3
"""
poisson distribution
"""


def fac(a):
    s = 1
    for i in range(1, a+1):
        s *= i
    return s

e = 2.7182818285


class Poisson:
    """
    poisson class
    """

    def __init__(self, data=None, lambtha=1.):
        """
        initializing the object
        """

        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = sum(data) / len(data)
        
        @classmethod
        def pmf(self, k):
            k = int(k)
            if k > len(self.data):
                return 0
            p = e**(-self.lambtha)*self.lambtha**k/fac(k)
            return p
