#!/usr/bin/env python3
"""NST that performs tasks for neural style transfer"""

import numpy as np
import tensorflow as tf


class NST:
    """
    Performs tasks for Neural Style Transfer
    """

    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image,
                 alpha=1e4, beta=1):
        if (type(style_image) is not np.ndarray or
                len(style_image.shape) != 3 or
                style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (type(content_image) is not np.ndarray or
                len(content_image.shape) != 3 or
                content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """Scales image dimensions and values to 0-1"""
        if (type(image) is not np.ndarray or
                len(image.shape) != 3 or
                image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, c = image.shape

        if h > w:
            h_new = 512
            w_new = int(w * (512 / h))
        else:
            w_new = 512
            h_new = int(h * (512 / w))

        new_shape = (h_new, w_new)

        image = np.expand_dims(image, axis=0)

        scaled_image = tf.image.resize(
            image, new_shape, method='bicubic'
        )
        scaled_image = tf.clip_by_value(
            scaled_image / 255, 0, 1
        )

        return scaled_image

    def load_model(self):
        """Creates the model used to calculate the style and content cost"""
        vgg = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet'
        )
        vgg.trainable = False

        inputs = vgg.input
        x = inputs
        
        # Lists to hold the specific tensors we want to output
        style_outputs = []
        content_output = None

        for layer in vgg.layers[1:]:
            # Replace Max Pooling with Average Pooling
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                x = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name
                )(x)
            else:
                x = layer(x)
            
            # Capture the outputs as we build the graph
            if layer.name in self.style_layers:
                style_outputs.append(x)
            if layer.name == self.content_layer:
                content_output = x

        model_outputs = style_outputs + [content_output]
        
        # Instantiate the model exactly once
        self.model = tf.keras.models.Model(inputs=inputs, outputs=model_outputs)
