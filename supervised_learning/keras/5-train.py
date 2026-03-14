#!/usr/bin/env python3
'''
training the data
'''


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, 
                epochs, verbose=True, shuffle=False, validation_data=None):
    '''
    network - the model to train
    data - ndarray of shape (m, nx) containing the input data
    labels - one-hot ndarray of chape (m, classes) containing labels of data
    batch_size - is the size of the batch used for mini-batch gradient descent
    epochs - the number of passes through data for mini-batch gradient descent
    verbose - the boolean that determines if output should be printed

    shuffle - a boolean that determines whether to shullfe batches every epoch
    Normally, it is a good idea to shuffle, but for reproducibility, 
    we have chosen to set the default to False

    returns: History object generated after training the model
    '''

    history = network.fit(
            x=data,
            y=labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2 if verbose else 0,
            validation_data=validation_data,
            shuffle=shuffle
    )
    print(history.history['accuracy'])
    return history
