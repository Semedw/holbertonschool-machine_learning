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

    if padding == 'same':
        h_out = int(np.ceil(h / sh))
        w_out = int(np.ceil(w / sw))

        ph = max(0, (h_out - 1) * sh + kh - h)
        pw = max(0, (w_out - 1) * sw + kw - w)

        top, left = ph // 2, pw // 2
        bottom, right = ph - top, pw - left

    elif padding == 'valid':
        top, bottom, left, right = 0, 0, 0, 0
        h_out = (h - kh) // sh + 1
        w_out = (w - kw) // sw + 1
    else:
        ph, pw = padding
        top, bottom, left, right = ph, ph, pw, pw
        h_out = (h + 2 * ph - kh) // sh + 1
        w_out = (w + 2 * pw - kw) // sw + 1

    padded_imgs = np.pad(images, ((0, 0), (top, bottom), (left, right)), 
                         mode='constant', constant_values=0)
    
    convolved = np.zeros((m, h_out, w_out))

    for i in range(h_out):
        for j in range(w_out):
            start_h = i * sh
            start_w = j * sw

            patch = padded_imgs[:,
                                start_h: start_h + kh,
                                start_w: start_w + kw]
            convolved[:, i, j] = np.sum(patch*kernel, axis=(1, 2))

    return convolved
