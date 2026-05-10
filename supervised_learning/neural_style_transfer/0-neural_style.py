#!/usr/bin/env python3
"""Neural Style Transfer"""

import numpy as np
import tensorflow as tf


class NST:
    """Performs Neural Style Transfer"""

    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1',
                    'block5_conv1']

    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Class constructor
        style_image is a tf.Tensor of shape (1, h, w, 3) containing the style image
        content_image is a tf.Tensor of shape (1, h, w, 3) containing the content image
        alpha is the weight for the content cost
        beta is the weight for the style cost"""

        if isinstance(style_image, np.ndarray) or style_image.shape[2] != 3 or len(style_image.shape) != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        if isinstance(content_image, np.ndarray) or content_image.shape[2] != 3 or len(content_image.shape) != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")
        if alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = style_image
        self.content_image = content_image
        self.alpha = alpha
        self.beta = beta
        self.model = self.load_model()
        self.style_features, self.content_feature = self.get_features()

    def scale_image(self, image):
        """Rescales an image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels
        image is a tf.Tensor of shape (h, w, 3) containing the image to scale
        Returns: the scaled image as a tf.Tensor of shape (1, h_new, w_new, 3)"""

        if not isinstance(image, np.ndarray) or np.ndarray(image).shape[2] != 3 or len(np.ndarray(image).shape) != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")

        max_side = max(image.shape[0], image.shape[1])
        scale = 512 / max_side
        new_height = int(image.shape[0] * scale)
        new_width = int(image.shape[1] * scale)
        resized_image = tf.image.resize(image, (new_height, new_width))
        scaled_image = resized_image / 255.0

        return tf.expand_dims(scaled_image, axis=0)
