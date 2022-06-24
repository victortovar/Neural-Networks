## Victor Tovar
## 1000874189
## Task 2

import tensorflow as tf
import numpy as np
import os
import sys
import time

def nn_dense_mnist():
    (layers, units_per_layer, rounds, hidden_activation) = invoked()
    if(layers < 2):
        print("Not enough layers...exiting")
        return
    dataset = tf.keras.datasets.mnist
    (training_set, test_set) = dataset.load_data()
    (training_images, training_labels) = training_set
    (test_images, test_labels) = test_set
    (training_number, rows, cols) = training_images.shape
    (test_number, rows, cols) = test_images.shape
    
    model = tf.keras.Sequential([])
    for i in range(2, layers):
        model.add(tf.keras.layers.Dense(units_per_layer, activation = hidden_activation))
    model.add(tf.keras.layers.Dense(10, activation = 'softmax'))
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    
    training_inputs = training_images.astype("float32")/255.0
    training_inputs = training_inputs.reshape(training_number, rows*cols)
    test_inputs = test_images.astype("float32")/255.0
    test_inputs = test_inputs.reshape(test_number, rows*cols)
    model.fit(training_inputs, training_labels, epochs = rounds)
    
    test_loss, test_acc = model.evaluate(test_inputs, test_labels, verbose=0)
    total_accuracy = 0.0
    accuracy = 0.0

    for i in range(0, len(test_inputs)):
        input_vector = test_inputs[i, :]
        input_vector = np.reshape(input_vector,(1,len(input_vector)))
        nn_output = model.predict(input_vector)
        nn_output = nn_output.flatten()
        predicted_class = np.argmax(nn_output)
        true_class = test_labels[i]
        (indices,) = np.nonzero(nn_output == nn_output[predicted_class])
        number_of_ties = np.prod(indices.shape)
        if(nn_output[true_class] == nn_output[predicted_class]):
            accuracy = 1.0/number_of_ties
        else:
            accuracy = 0.0
        total_accuracy = total_accuracy + accuracy
        print("ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f\n" %
        (i + 1, predicted_class, true_class, accuracy))
    print("Classification Accuracy: %6.4f%%" %(total_accuracy*100/len(test_inputs)))

    return


def invoked():
    argc = len(sys.argv)
    if(argc > 5):
        print("Too many arguments")
        return
    if(argc == 5):
        layers = int(sys.argv[1])
        units_per_layer = int(sys.argv[2])
        rounds = int(sys.argv[3])
        hidden_activation = sys.argv[4]        
    return(layers, units_per_layer, rounds, hidden_activation)  


nn_dense_mnist()
