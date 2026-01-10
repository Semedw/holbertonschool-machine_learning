#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def line():

    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    # your code here
    x = np.arange(0, 11)
    plt.plot(x, y, color='#FF0000')
    plt.xlim(0, 10)
    plt.xticks(np.arange(0, 11, 2))
    plt.show()
