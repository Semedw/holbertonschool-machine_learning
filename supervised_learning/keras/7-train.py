#!/usr/bin/env python3
'''
training the data
'''


import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.01, decay_rate=1, save_best=False,
                filepath=None):
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

    callbacks = []

    if validation_data and early_stopping:
        early_stop = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            verbose=verbose
        )
        callbacks.append(early_stop)

    if learning_rate_decay and validation_data:

        def scheduler(epoch):
            '''
            implement inverse time decay
            '''
            return alpha / (1 + decay_rate * epoch)

        callbacks.append(K.callbacks.LearningRateScheduler(
            scheduler,
            verbose=1
        ))

    if save_best:
        callbacks.append(K.callbacks.ModelCheckpoint(filepath=filepath))
    
    optimizer = K.optimizers.SGD(learning_rate=alpha)

    network.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = network.fit(
                x=data,
                y=labels,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1 if verbose else 0,
                callbacks=callbacks,
                validation_data=validation_data,
                shuffle=shuffle
        )

    return history
