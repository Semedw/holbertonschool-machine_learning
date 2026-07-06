#!/usr/bin/env python3
"""A module that does the trick"""
import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a trained gensim Word2Vec model to a trainable Keras
    Embedding layer.
    """
    weights = model.wv.vectors

    embedding = tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True
    )

    return embedding
from gensim.test.utils import common_texts
word2vec_model = __import__('2-word2vec').word2vec_model
gensim_to_keras = __import__('3-gensim_to_keras').gensim_to_keras

print(common_texts[:2])
w2v = word2vec_model(common_texts, min_count=1)
print(gensim_to_keras(w2v))
