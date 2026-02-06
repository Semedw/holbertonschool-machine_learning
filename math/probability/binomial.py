#!/usr/bin/env python3
"""
initializing binomial
"""


pi = 3.1415926536


def fac(x):
    s = 1
    for i in range(1, x+1):
        s *= i
    return s


class Binomial:
    """
    inside the binomial class
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        initializing the data
        """
        self.data = data
        self.n = n
        self.p = p
        if data is None:
            if n <= 0:
                raise ValueError('n must be a positive value')
            if not (p > 0 and p < 1):
                raise ValueError('p must be greater than 0 and less than 1')
        else:
            if n <= 0:
                raise ValueError('n must be a positive value')
            if not (p > 0 and p < 1):
                raise ValueError('p must be greater than 0 and less than 1')
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = sum(data) / len(data)
            variance = sum((x-mean)**2 for x in data) / len(data)
            p = 1 - (variance / mean)
            self.n = int(round(mean/p))
            self.p = float(mean/self.n)

    def pmf(self, k):
        """
        calculating pmf
        """
        k = int(k)
        if k < 0:
            return 0
        res = fac(self.n)/(fac(k)*fac(self.n-k))*(self.p**k)*((1-self.p)**(self.n-k))
        return res
