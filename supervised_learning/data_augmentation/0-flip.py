#!/usr/bin/env python3
'''
flip the image
'''

import tensorflow as tf


def flip_image(image):
    '''
    image - 3D tf.Tensor containing the image
    '''
    return tf.image.flip_left_right(
            image
            )
