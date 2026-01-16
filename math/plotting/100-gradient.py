#!/usr/bin/env python3
"""
gradient
"""

import numpy as np
import matplotlib.pyplot as plt


def gradient():
    """
    making colorbar
    """
    np.random.seed(5)

    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))
    plt.figure(figsize=(6.4, 4.8))

    # your code here
    data = z.reshape(40, 50) 
    plt.scatter(x, y, c=z, cmap='viridis')
    im = plt.imshow(data, cmap='viridis')
    im.set_visible(False)
    plt.colorbar(im)
    plt.xlabel('x coordinates (m)')
    plt.xticks(np.arange(-30, 31, 10))
    plt.xlim(-40, 40)
    plt.ylabel('y coordinates (m)')
    plt.yticks(np.arange(30, -31, -10))
    plt.ylim(-40, 40)
    plt.title('Mountain Elevation')
    plt.show()

gradient()
