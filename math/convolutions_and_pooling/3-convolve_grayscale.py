#!/usr/bin/env python3
'''
strided convolution
'''

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1,1)):
    '''
    images - a numpy.ndarray with shape (m, h, w) containing multiple grayscale images
        m - the number of images
        h - the height in pixels of the images
        w - the width in pixels of the images

    kernel - a numpy.ndarray with shape (kh, kw) containing the kernel for the convolution
        kh - the height of the kernel
        kw - the width of the kernel

    padding - either a tuple of (ph, pw), 'same', or 'valid'
        if 'same', performs a same convolution
        if 'valid', performs a valid convolution
        if a tuple:
            ph - the padding for the height of the image
            pw - the padding for the wis

    stride - a tuple of (sh, sw)

    Returns: a numpy.ndarray containing the convolved images
    '''
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if type(padding) is tuple:
        ph, pw = padding
    elif padding == 'same':
        ph = int(((h - kh) / sh + 1 - 1) / 2) + 1
        pw = int(((w - kw) / sw + 1 - 1) / 2) + 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    
    # Pad the images
    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')
    # Calculate the dimensions of the output
    h_out = int(((h + 2 * ph - kh) / sh) + 1)
    w_out = int(((w + 2 * pw - kw) / sw) + 1)
    # Initialize the output array    
    output = np.zeros((m, h_out, w_out))
    # Perform the convolution
    for i in range(h_out):
        for j in range(w_out):
            # Define the region of interest
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw
            # Extract the region of interest and perform element-wise multiplication with the kernel
            output[:, i, j] = np.sum(images_padded[:, h_start:h_end, w_start:w_end] * kernel, axis=(1, 2))
    return output
