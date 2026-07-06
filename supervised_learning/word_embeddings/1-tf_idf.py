#!/usr/bin/env python3
"""A module that does the trick"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """A function that does the trick"""
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    embeddings = X.toarray()
    features = vectorizer.get_feature_names_out()
    return embeddings, features
