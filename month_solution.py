## Victor Tovar
## 1000874189
## Assignment 6

import numpy as np
from tensorflow import keras
from sklearn.metrics import confusion_matrix

import math


def data_normalization(raw_data, train_start, train_end):
    (number, dimensions) = raw_data.shape
    normalized_data = np.zeros((number, dimensions))
    for d in range(0, dimensions):
        feature_values = raw_data[train_start:train_end, d]
        m = np.mean(feature_values)
        s = np.std(feature_values)
        normalized_data[:,d] = (raw_data[:,d] - m)/s
    return normalized_data

def make_inputs_and_targets(data, months, size, sampling):
    length = 14*24*6
    target_time = 24*6
    (data_length, dimensions) = data.shape
    input_length = math.ceil(length/sampling)
    inputs = np.zeros((size, input_length, dimensions))
    targets = np.zeros((size))
    for i in range(0, size):
        (input, target) = random_input(data, length, target_time, months, sampling)
        inputs[i] = input
        targets[i] = target
    return (inputs, targets)

def random_input(data, length, target_time, months, sampling):
    (data_length, dimensions) = data.shape
    max_start = data_length - length - target_time
    start = np.random.randint(0, max_start)
    end = start + length
    result_input = data[start:end:sampling, :]
    target = months[end+target_time - 1]
    return (result_input, target)

    
def build_and_train_dense(train_inputs, train_targets, val_inputs, val_targets, filename):
    model = keras.Sequential([keras.Input(shape=train_inputs[0].shape), 
                              keras.layers.Flatten(),
                              keras.layers.Dense(64, activation="tanh"),
                              keras.layers.Dropout(0.2),
                              keras.layers.Dense(32, activation="tanh"),
                              keras.layers.Dense(12),
                              keras.layers.Dense(12)])
    model.compile(optimizer="adam", loss="mae", metrics=["accuracy"])
    callbacks = [keras.callbacks.ModelCheckpoint(filename, save_best_only=True)]
    history_dense = model.fit(train_inputs, train_targets, epochs=2, 
                          validation_data=(val_inputs, val_targets),
                          callbacks=callbacks)

    return history_dense

def build_and_train_rnn(train_inputs, train_targets, val_inputs, val_targets, filename):
    model = keras.Sequential([keras.Input(shape=train_inputs[0].shape),
                              keras.layers.Bidirectional(keras.layers.LSTM(32)),
                              keras.layers.Flatten(),
                            #   keras.layers.Bidirectional(keras.layers.LSTM(20)),
                            #   keras.layers.Flatten(),
                              keras.layers.Dense(16),
                              keras.layers.Dense(12)])
    model.compile(optimizer="adam", loss="mae", metrics=["accuracy"])
    callbacks = [keras.callbacks.ModelCheckpoint(filename, save_best_only=True)]
    history_rnn = model.fit(train_inputs, train_targets, epochs=1,
                            validation_data=(val_inputs, val_targets),
                            callbacks=callbacks)

    return history_rnn

def test_model(filename, test_inputs, test_targets):
    model = keras.models.load_model(filename)
    (loss, test_acc) = model.evaluate(test_inputs, test_targets)
    return test_acc

def confusion_matrix(filename, test_inputs, test_targets):
    model = keras.models.load_model(filename)
    input_shape = (test_inputs[0].shape)
    (a, b) = input_shape
    classes = np.arange(start=0, stop=12, step=1)
    conf_matrix = np.zeros((len(classes), len(classes)))
    predicted = np.zeros(len(test_targets))
    for i in range(0, len(test_inputs)):
        input_vector = test_inputs[i, :]
        input_vector = np.reshape(input_vector,(1, a, b))
        nn_output = model.predict(input_vector)
        predicted_class = np.argmax(nn_output)
        predicted[i] = predicted_class
        # nn_output = nn_output.flatten()
        # true_class = test_targets[i]
        # (indices,) = np.nonzero(nn_output == nn_output[predicted_class])
    for i in range(len(classes)):
        for j in range(len(classes)):
            conf_matrix[i][j] = np.sum((test_targets == classes[i]) & (predicted == classes[j]))
   
    return conf_matrix

