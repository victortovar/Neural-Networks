## Victor Tovar
## 1000874189
## Task 1

import tensorflow as tf
import numpy as np
import os
import sys
import time

def nn_keras():
    (training_file, test_file, layers, units_per_layer, rounds, hidden_activation) = invoked()
    (training_set, test_set) = read_uci1(training_file, test_file)
    (training_inputs, training_labels) = training_set
    (test_inputs, test_labels) = test_set
    max_value = np.max(np.abs(training_inputs))
    training_inputs = training_inputs/max_value
    test_inputs = test_inputs/max_value
    input_shape = training_inputs[0].shape
    number_of_classes = np.max([np.max(training_labels), np.max(test_labels)]) + 1
    model = tf.keras.Sequential([tf.keras.Input(shape = input_shape)])
    for i in range(2, layers):
        model.add(tf.keras.layers.Dense(units_per_layer, activation = hidden_activation))
    model.add(tf.keras.layers.Dense(number_of_classes, activation = 'softmax'))
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
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


def read_uci_file(pathname, labels_to_ints, ints_to_labels):
    if not(os.path.isfile(pathname)):
        print("read_data: %s not found", pathname)
        return None

    in_file = open(pathname, encoding='utf-8-sig')
    file_lines = in_file.readlines()
    in_file.close()

    rows = len(file_lines)
    if (rows == 0):
        print("read_data: zero rows in %s", pathname)
        return None
        
    
    cols = len(file_lines[0].split())
    data = np.zeros((rows, cols-1))
    labels = np.zeros((rows,1))
    for row in range(0, rows):
        line = file_lines[row].strip()
        items = line.split()
        if (len(items) != cols):
            print("read_data: Line %d, %d columns expected, %d columns found" %(row, cols, len(items)))
            return None
        for col in range(0, cols-1):
            data[row][col] = float(items[col])
        
        # the last column is a string representing the class label
        label = items[cols-1]
        if (label in labels_to_ints):
            ilabel = labels_to_ints[label]
        else:
            ilabel = len(labels_to_ints)
            labels_to_ints[label] = ilabel
            ints_to_labels[ilabel] = label
        
        labels[row] = ilabel

    labels = labels.astype(int)
    return (data, labels)


def read_uci_dataset(training_file, test_file):
    # training_file = directory + "/" + dataset_name + "_training.txt"
    # test_file = directory + "/" + dataset_name + "_test.txt"

    labels_to_ints = {}
    ints_to_labels = {}

    (train_data, train_labels) = read_uci_file(training_file, labels_to_ints, ints_to_labels)
    (test_data, test_labels) = read_uci_file(test_file, labels_to_ints, ints_to_labels)
    return ((train_data, train_labels), (test_data, test_labels), (ints_to_labels, labels_to_ints))


def read_uci1(training_file, test_file):
    # training_file = directory + "/" + dataset_name + "_training.txt"
    # test_file = directory + "/" + dataset_name + "_test.txt"

    labels_to_ints = {}
    ints_to_labels = {}

    (train_data, train_labels) = read_uci_file(training_file, labels_to_ints, ints_to_labels)
    (test_data, test_labels) = read_uci_file(test_file, labels_to_ints, ints_to_labels)
    return ((train_data, train_labels), (test_data, test_labels))

def invoked():
    argc = len(sys.argv)
    if(argc < 5):
        print("Error:  Too few arguments")
        return
    elif(argc > 7):
        print("Error:  Too many arguments")
        return
    else:
        training_file = sys.argv[1]
        test_file = sys.argv[2]
        layers = int(sys.argv[3])
        if(layers < 2):
            print("Not enough layers")
            return
        if(argc == 5):
            units_per_layer = 1
            rounds = int(sys.argv[4])
            hidden_activation = None
        if(argc == 6):
            units_per_layer = 10
            rounds = int(sys.argv[4])
            hidden_activation = sys.argv[5]
        if(argc == 7):
            units_per_layer = int(sys.argv[4])
            rounds = int(sys.argv[5])
            hidden_activation = sys.argv[6]

        return (training_file, test_file, layers, units_per_layer, rounds, hidden_activation)

nn_keras()