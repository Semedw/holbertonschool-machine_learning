#!/usr/bin/env python3
"""
initializing binomial
"""


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
            if n < 0:
                raise ValueError('n must be a positive value')
            if not(p>=0 and p<=1):
                raise ValueError('p must be greater than 0 and less than 1')
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = sum(data) / len(data)
            variance = sum((x-mean)**2 for x in data) / len(data)
            p = 1 - (variance / mean)
            self.n = int(round(mean/p))
            self.p = float(mean/n)
