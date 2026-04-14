#!/usr/bin/env python3
'''
Inception Block
'''

from tensorflow import keras as K


def inception_block(A_prev, filters):
    '''
    A_prev - the output from the previous layer
    filters - a tuple or list containing F1, F3R, F3,F5R, F5, FPP, respectively:
        F1 - the number of filters in the 1x1 convolution
        F3R - the number of filters in the 1x1 convolution before the 3x3 convolution
        F3 - the number of filters in the 3x3 convolution
        F5R - the number of filters in the 1x1 convolution before the 5x5 convolution
        F5 - the number of filters in the 5x5 convolution
        FPP - the number of filters in the 1x1 convolution after the max pooling

    All convolutions inside the inception block should use a rectified linear activation (ReLU)
    Returns: the concatenated output of the inception block
    '''

    F1, F3R, F3, F5R, F5, FPP = filters

    branch1 = K.layers.Conv2D(F1, (1, 1), padding='same', activation='relu')(A_prev)

    branch2 = K.layers.Conv2D(F3R, (1, 1), padding='same', activation='relu')(A_prev)
    branch2 = K.layers.Conv2D(F3, (3, 3), padding='same', activation='relu')(branch2)

    branch3 = K.layers.Conv2D(F5R, (1, 1), padding='same', activation='relu')(A_prev)
    branch3 = K.layers.Conv2D(F5, (5, 5), padding='same', activation='relu')(branch3)

    branch4 = K.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same', activation='relu')(A_prev)
    branch4 = K.layers.Conv2D(FPP, (1, 1), padding='same', activation='relu')(branch4)

    output = K.layers.Concatenate(axis=-1)([branch1, branch2, branch3, branch4])

    return output
