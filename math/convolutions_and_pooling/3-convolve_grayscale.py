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
    '''

    m, h, w = images.shape
    kh, kw = kernel.shape

    sh, sw = stride

    if isinstance(padding, tuple):
        ph, pw = padding
    
    if padding == 'valid':
        h_out = h - kh + 1
        w_out = w - kw + 1

        convolved = np.zeros((m, h_out, w_out))

        for i in range(0, h_out, sh):
            for j in range(0, w_out, sw):
                image = images[:, i: i + kh, j: j + kw]
                convolved[:, i, j] = np.sum(image*kernel, axis=(1, 2))
        return convolved
    
    elif padding == 'same':
        ph = kh // 2
        pw = kw // 2

        images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                            mode='constant', constant_values=0)

        convolved = np.zeros((m, h, w))

        for i in range(0, h, sh):
            for j in range(0, w, sw):
                image = images_padded[:, i: i + kh, j: j + kw]
                convolved[:, i, j] = np.sum(image*kernel, axis=(1, 2))
        return convolved
    
    else:
        h_out = (h + 2 * ph) - kh + 1
        w_out = (w + 2 * pw) - kw + 1

        images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                            mode='constant', constant_values=0)

        convolved = np.zeros((m, h_out, w_out))

        for i in range(0, h_out, sh):
            for j in range(0, w_out, sw):
                image = images_padded[:, i: i + kh, j: j + kw]
                convolved[:, i, j] = np.sum(image*kernel, axis=(1, 2))
        return convolved
