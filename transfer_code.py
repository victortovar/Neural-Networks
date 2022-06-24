## Victor Tovar
## 1000874189
## Task 1


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys

##Trains the given model given the inputs and the labels
def train_model(model, cifar_tr_inputs, cifar_tr_labels, batch_size, epochs):
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    model.fit(cifar_tr_inputs, cifar_tr_labels, epochs = epochs, batch_size = batch_size)
    return model

def load_and_refine(filename, training_inputs, training_labels, batch_size, epochs):
    model = tf.keras.models.load_model(filename)
    num_layers = len(model.layers)
    small_num_classes = np.max(training_labels) + 1    
    refined_model = keras.models.Sequential()
    for layer in model.layers[0:-1]:
        refined_model.add(layer)
    for i in range(0, num_layers-1):
        refined_model.layers[i].trainable = False
    refined_model.add(layers.Dense(small_num_classes, activation="softmax"))
    refined_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=["accuracy"])
    training_inputs = np.expand_dims(training_inputs, -1)
    training_labels = np.expand_dims(training_labels, -1)
    temp = training_inputs
    (num, _,_,_) = temp.shape
    temp = np.repeat(temp, 3, axis=3)
    training_inputs = np.zeros((num, 32, 32, 3))
    training_inputs[:,2:30, 2:30, :] = temp
    refined_model.fit(training_inputs, training_labels, epochs=epochs, batch_size=batch_size)
    return refined_model

def evaluate_my_model(model, test_inputs, test_labels):
    test_inputs = np.expand_dims(test_inputs, -1)
    temp = test_inputs
    (num, _,_,_) = temp.shape
    temp = np.repeat(temp, 3, axis=3)
    test_inputs = np.zeros((num, 32, 32,3))
    test_inputs[:,2:30, 2:30, :] = temp
    _, test_acc = model.evaluate(test_inputs, test_labels, verbose=0)
    return test_acc