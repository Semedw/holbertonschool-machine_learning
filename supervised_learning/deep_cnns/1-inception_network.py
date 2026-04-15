#!/usr/bin/env python3
'''
inception network
'''

from tensorflow import keras as K

inception_block = __import__('0-inception_block').inception_block


def inception_network():
    '''
    You can assume the input data will have shape (224, 224, 3)
    All convolutions inside and outside the inception block should
    use a rectified linear activation (ReLU)

    Returns: the keras model
    '''

    X = K.Input(shape=(224, 224, 3))
 
    # ── Stem ────────────────────────────────────────────────────────────────
    # Conv 7×7 / stride 2  →  112×112×64
    Y = K.layers.Conv2D(
        64, kernel_size=7, strides=2, padding='same', activation='relu'
    )(X)
 
    # MaxPool 3×3 / stride 2  →  56×56×64
    Y = K.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(Y)
 
    # Conv 1×1  →  56×56×64
    Y = K.layers.Conv2D(
        64, kernel_size=1, strides=1, padding='same', activation='relu'
    )(Y)
 
    # Conv 3×3  →  56×56×192
    Y = K.layers.Conv2D(
        192, kernel_size=3, strides=1, padding='same', activation='relu'
    )(Y)
 
    # MaxPool 3×3 / stride 2  →  28×28×192
    Y = K.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(Y)
 
    # ── Inception 3a  (28×28)  ───────────────────────────────────────────────
    # filters: [#1x1, #3x3_reduce, #3x3, #5x5_reduce, #5x5, #pool_proj]
    Y = inception_block(Y, [64, 96, 128, 16, 32, 32])
 
    # ── Inception 3b  (28×28)  ───────────────────────────────────────────────
    Y = inception_block(Y, [128, 128, 192, 32, 96, 64])
 
    # MaxPool 3×3 / stride 2  →  14×14×480
    Y = K.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(Y)
 
    # ── Inception 4a  (14×14)  ───────────────────────────────────────────────
    Y = inception_block(Y, [192, 96, 208, 16, 48, 64])
 
    # ── Inception 4b  (14×14)  ───────────────────────────────────────────────
    Y = inception_block(Y, [160, 112, 224, 24, 64, 64])
 
    # ── Inception 4c  (14×14)  ───────────────────────────────────────────────
    Y = inception_block(Y, [128, 128, 256, 24, 64, 64])
 
    # ── Inception 4d  (14×14)  ───────────────────────────────────────────────
    Y = inception_block(Y, [112, 144, 288, 32, 64, 64])
 
    # ── Inception 4e  (14×14)  ───────────────────────────────────────────────
    Y = inception_block(Y, [256, 160, 320, 32, 128, 128])
 
    # MaxPool 3×3 / stride 2  →  7×7×832
    Y = K.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(Y)
 
    # ── Inception 5a  (7×7)  ─────────────────────────────────────────────────
    Y = inception_block(Y, [256, 160, 320, 32, 128, 128])
 
    # ── Inception 5b  (7×7)  ─────────────────────────────────────────────────
    Y = inception_block(Y, [384, 192, 384, 48, 128, 128])
 
    # ── Classifier ───────────────────────────────────────────────────────────
    # AvgPool 7×7 / stride 1  →  1×1×1024
    Y = K.layers.AveragePooling2D(pool_size=7, strides=1, padding='valid')(Y)
 
    # Dropout 40 %
    Y = K.layers.Dropout(rate=0.4)(Y)
 
    # Flatten  →  1024
    Y = K.layers.Flatten()(Y)
 
    # Softmax  →  1000
    Y = K.layers.Dense(1000, activation='softmax')(Y)
 
    model = K.Model(inputs=X, outputs=Y)
    
    return model
 