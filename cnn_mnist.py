## Victor Tovar
## 1000874189
## Task 3

import tensorflow as tf
import numpy as np
import os
import sys
import time

def invoked():
    argc = len(sys.argv)
    if(argc > 7):
        print("Too many arguments")
        return
    if(argc == 7):
        blocks = int(sys.argv[1])
        filter_size = int(sys.argv[2])
        filter_number = int(sys.argv[3])
        region_size = int(sys.argv[4])
        rounds = int(sys.argv[5])
        cnn_activation = sys.argv[6]    
    return(blocks, filter_size, filter_number, region_size, rounds, cnn_activation)

def cnn_mnist():
    dataset = tf.keras.datasets.mnist
    (blocks, filter_size, filter_number, region_size, rounds, cnn_activation) = invoked()
    (x_train, train_labels), (x_test, test_labels) = dataset.load_data()
    number_of_classes = np.max([np.max(train_labels), np.max(test_labels)]) + 1

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    input_shape = x_train[0].shape

    model = tf.keras.Sequential([tf.keras.Input(shape=input_shape)])
    for i in range(0, blocks):
        model.add(tf.keras.layers.Conv2D(filter_number, kernel_size=(filter_size, filter_size), activation=cnn_activation))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(region_size, region_size)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(number_of_classes, activation='softmax'))
    #model.summary()
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    model.fit(x_train, train_labels, epochs = rounds)
    
    test_loss, test_acc = model.evaluate(x_test, test_labels, verbose=0)
    total_accuracy = 0.0
    accuracy = 0.0
    for i in range(0, len(x_test)):
        input_vector = x_test[i, :]
        input_vector = np.reshape(input_vector,(1, 28, 28, 1))
        nn_output = model.predict(input_vector)
        predicted_class = np.argmax(nn_output)
        nn_output = nn_output.flatten()
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
    print("Classification Accuracy: %6.4f%%" %(total_accuracy*100/len(x_test)))

    return


cnn_mnist()