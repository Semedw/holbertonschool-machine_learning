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
        self.gram_style_features = self.generate_features()[0]
        self.content_feature = self.generate_features()[1]

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
        self.model = tf.keras.models.Model(inputs=inputs,
                                           outputs=model_outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """Calculates the gram matrix of an input layer"""
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
                len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        channels = int(input_layer.shape[-1])
        batch_size = tf.shape(input_layer)[0]
        a = tf.reshape(input_layer, [batch_size, -1, channels])
        n = tf.shape(a)[1]
        gram = tf.matmul(a, a, transpose_a=True)
        return gram / tf.cast(n, input_layer.dtype)

    def generate_features(self):
        """Extracts the features used to calculate the style and
        content cost"""
        preprocessed_style = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255)
        preprocessed_content = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255)

        style_outputs = self.model(preprocessed_style)[:-1]
        content_output = self.model(preprocessed_content)[-1]

        style_features = [self.gram_matrix(style_output)
                          for style_output in style_outputs]

        content_features = content_output

        self.gram_style_features = style_features
        self.content_feature = content_features

        return self.gram_style_features, self.content_feature

    def layer_style_cost(self, style_output, gram_target):
        """Calculates the style cost for a single layer"""
        _, c, _ = gram_target.shape
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or \
                len(style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or \
                len(gram_target.shape) != 3:
            raise TypeError(f"gram_target must be a tensor of shape [1, {c}, {c}]")

        gram_style = self.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_style - gram_target))
