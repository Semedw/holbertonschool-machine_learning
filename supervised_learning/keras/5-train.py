#!/usr/bin/env python3
'''
training the model
'''


import tensorflow.keras as K


def train_model(network, data, labels, 
                batch_size, epochs, verbose=True,
                shuffle=False, validation_data=None):
    '''
    trains a model using mini-batch gradient descent
    network is the model to train
    data is the input data
    labels are the one-hot labels of data
    batch_size is the number of data points in a batch
    epochs is the number of times the training should pass through
           the whole dataset
    verbose is a boolean that determines if output should be printed
            during training
    shuffle is a boolean that determines whether to shuffle the batches
    every epoch. Normally, it is a good idea to shuffle, but for
    reproducibility, we have chosen to set the default to False.
    Returns: the History object generated after training the model
    '''
    return network.fit(x=data, y=labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data)
