#!/usr/bin/env python3
"""
initial exponential
"""


e = 2.7182818285


def fac(x):
    """
    calculating factorial
    """
    s = 1
    for i in range(1, x+1):
        s *= i
    return s


class Exponential:
    """
    exponential class
    """

    def __init__(self, data=None, lambtha=1.):
        """
        initializing the class
        """
        self.data = data
        self.lambtha = lambtha
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = lambtha
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = 1 / (sum(data) / len(data))
            if lambtha < 0:
                raise ValueError('lambtha must be a positive value')

    def pdf(self, x):
        """
        calculating pdf of exponential in given time period
        x - time period
        """
        res = self.lambtha*e**(-self.lambtha*x)
        return res
