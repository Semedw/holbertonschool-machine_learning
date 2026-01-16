#!/usr/bin/env python3
"""
salam necesiz
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

data = np.load("data.npy")
labels = np.load('labels.npy')

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

# your code here
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

colors = (['blue'] * 50 + ['red'] * 50 + ['yellow'] * 50)
ax.scatter(pca_data[ :150, 0], pca_data[ :150, 1], pca_data[ :150, 2],
           c=colors) 
ax.set_xlabel('U1')
ax.set_ylabel('U2')
ax.set_zlabel('U3')
ax.set_title('PCA of Iris Dataset')
plt.show()
