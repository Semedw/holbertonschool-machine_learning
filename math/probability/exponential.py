#!/usr/bin/env python3
"""
initial exponential
"""


e = 2.7182818285


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
