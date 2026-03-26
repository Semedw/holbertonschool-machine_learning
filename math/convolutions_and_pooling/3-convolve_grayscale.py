#!/usr/bin/env python3
"""
convolve_grayscale
"""
import numpy as np


def ceil(a):
    '''
    ceil function that rounds a up to the nearest integer.
    '''
    b = a // 1
    if a != b:
        return int(b + 1)
    return int(a)


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images with custom padding.

    images: A numpy.ndarray with shape (m, h, w) containing multiple grayscale
        images.
        - m is the number of images.
        - h is the height in pixels of the images.
        - w is the width in pixels of the images.
    kernel: A numpy.ndarray with shape (kh, kw) containing the kernel of the
        convolution.
        - kh is the height of the kernel.
        - kw is the width of the kernel.
    padding: Either a tuple of (ph, pw), 'same', or 'valid'.
        - If 'same', performs a same convolution
        - If 'valid', performs a valid convolution
        - If a tuple:
            - ph is the padding on the height of the image.
            - pw is the padding on the width of the image.
    stride: A tuple of (sh, sw).
        - sh is the stride across the height of the image.
        - sw is the stride across the width of the image.

    Returns: A numpy.ndarray containing the convolved images.
    """

    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = max((h - 1) * sh + kh - h, 0) // 2
        pw = max((w - 1) * sw + kw - w, 0) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')

    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2, 3))

    return output
