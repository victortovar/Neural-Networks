#Victor Tovar
#1000874189


import numpy as np
import os
import sys
import time
from random import seed
from random import random


def neural_network():
    argc = len(sys.argv)
    if(argc < 5):
        print("Error:  Too few arguments")
        return
    elif(argc > 6):
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
            units_per_layer = 10
            rounds = int(sys.argv[4])
        if(argc == 6):
            units_per_layer = int(sys.argv[4])
            rounds = int(sys.argv[5])
    string_class_labels = []
    (training_data, t_training, string_class_labels, input_size, not_used) = load_and_read_files(training_file, string_class_labels)
    (testing_data, t_testing, string_class_labels, unused, q) = load_and_read_files(test_file, string_class_labels)
    max = get_absolute_max(training_data)
    normalize_data(training_data, max)
    normalize_data(testing_data, max)
    weights = randomize_weights(input_size, layers, units_per_layer, len(string_class_labels) - 1)
    biases = randomize_biases(layers, units_per_layer, len(string_class_labels)-1)
    ## Trains the network for the specified number of rounds
    for r in range(0,rounds):
        N = 0.98**r
        for n in range(1, len(training_data)):
            z = initialize_input_layer(training_data[n])
            N = 1
            train_network(z, biases, weights, layers, units_per_layer, t_training[n], len(string_class_labels), N)
    test_network(testing_data, weights, biases, layers, units_per_layer, t_testing, string_class_labels, q)
## tests the trained neural network for accuracy 
def test_network(testing_data, weights, biases, layers, units_per_layer, t_testing, string_class_labels, q):
    total_accuracy = 0.0
    for object_id in range(1, len(testing_data)):
        true_class = q[object_id-1]
        z = compute_output(testing_data[object_id], weights, biases, units_per_layer, t_testing[object_id])
        accuracy, predicted_class = calculate_accuracy(z[layers], t_testing[object_id], string_class_labels)
        total_accuracy = total_accuracy + accuracy
        print("ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f\n" %(object_id, predicted_class, true_class, accuracy))
    classification_accuracy = total_accuracy/len(testing_data)
    print("Classification accuracy= %6.4f\n" % (classification_accuracy))
    return

def calculate_accuracy(z, t, string_class_labels):
    max = 1
    accuracy = 0
    for i in range(2, len(t)):
        if(z[i] > z[max]):
            max = i
    if(t[max] == 1):
        accuracy = 1
    else:
        accuracy = 0
    predicted_class = string_class_labels[max]
    return (accuracy, predicted_class)
## Computes the output of each input given for the trained neural network
def compute_output(input, weights, biases, units_per_layer, t):
    z = [None, input]
    for layer in range(2, len(biases)):
        z.append([None])
        for i in range(1, len(weights[layer])):
            a = biases[layer][i] + weighted_sum(weights[layer][i], z[layer-1]) 
            z[layer].append(sigmoid(a))
    return z

## Trains neural network by using various built in functions
def train_network(z, biases, weights, layers, units_per_layer, t, last_layer_size, N):
    a = [None, None]
    for layer in range(2, layers+1):
        a.append([None])
        if(layer == layers):
            cols = last_layer_size-1
        else:
            cols = units_per_layer
        for i in range(1, cols + 1):
            a[layer].append(biases[layer][i] + weighted_sum(weights[layer][i], z[layer-1]))
        z = update_z(z, a[layer], layer)
    delta = compute_new_deltas(z, t, layers, last_layer_size, weights)
    update_weights(weights, delta, z, layers, biases, N)
    return

# Updates the values of the weights according to the delta values
# inputs include the weight array, delta array, total layers in network,
# the biases array, and the N value used to adjust weights and biases
def update_weights(weights, delta, z, layers, biases, N):
    for l in range(2, layers + 1): ##(2, layers + 1)  
        for i in range(1, len(z[l])):
            biases[l][i] = biases[l][i] - N * delta[l][i]
            for j in range(1, len(z[l-1])):
                weights[l][i][j] = weights[l][i][j] - N *delta[l][i]*z[l-1][j]
    return

#computes the new delta values to be used to update the weights
def compute_new_deltas(z, t, layers, last_layer_size, weights):
    delta = [None, None]
    for layer in range(2, layers+1):
        delta.append([])
    delta[layers].append(None)
    for i in range(1, last_layer_size):
        delta[layers].append((z[layers][i]- t[i])*z[layers][i]*(1-z[layers][i]))
    for l in range(layers - 1,1, -1):
        delta[l].append(None)
        for i in range(1, len(z[l])):
            delta[l].append(delta_summation(delta[l+1], weights[l+1], len(z[l+1]), i) * z[l][i]*(1 - z[l][i]))
    return delta

## Computes the summation to be used for for calculating delta values by using
## values of the next layer
def delta_summation(delta, weights, j, i):
    sum = 0.0
    for k in range(1, j):
        sum = sum + (delta[k] * weights[k][i])
    return sum

#updates the z matrix that stores the inputs for the next layer in the network
def update_z(z, a, layer):
    z.append([None])
    for i in range(1, len(a)):
        z[layer].append(sigmoid(a[i])) 
    return z

#Computes the weighted sum of the weights and the inputs
def weighted_sum(weights, z):
    sum = 0.0
    for j in range(1, len(z)):
        sum = sum + (weights[j]*z[j])
    return sum

#returns the sigmoid of the 'a' value passed in
def sigmoid(a):
    e = np.exp(-a)
    sig = 1/(1+e)
    return sig
     
# creates input array 'z' to be used to calculate outputs
def initialize_input_layer(data):
    z = [None]
    z.append([None])
    for i in range(1, len(data)):
        z[1].append(data[i])
    return z

#generates random biases and adds them to a 2D array
def randomize_biases(layers, units_per_layer, last_layer_size):
    biases = [None, None]
    seed()
    for layer in range(2, layers + 1):
        biases.append([])
        if(layer < last_layer_size):
            cols = units_per_layer
        else:
            cols = last_layer_size
        for b in range(0, cols + 1):
            if(layer < 2 or b < 1):
                biases[layer].append(None)
            else:
                biases[layer].append(0.1 * random() - 0.05)
    return biases
#generates random weights and adds them to a 3D array
def randomize_weights(input_size, layers, units_per_layer, last_layer_size):
    weights = [None, None]
    seed()
    cols = input_size
    for layer in range(2, layers + 1):
        weights.append([])
        if(layer == layers):
            end = last_layer_size
        else:
            end = units_per_layer
        for row in range(0, end+1):
            weights[layer].append([])
            if(layer > 2):
                cols = units_per_layer
            for col in range(0, cols + 1):
                if(col < 1 or row < 1):
                    weights[layer][row].append(None)
                else:
                    weights[layer][row].append(0.1 * random() - 0.05)
    return weights

#normalizes the data given 
def normalize_data(data, max):
    for i in range(1, len(data)):
        for j in range(len(data[i])):
            if data[i][j] is not None:
                data[i][j] = data[i][j]/max
    return data

#Gets the absolute maximum of the data to be used for normalizing
def get_absolute_max(data):
    max = -1
    for i in range(1, len(data)):
        for j in range(1, len(data[i])):
            if data[i][j] is not None:
                if(data[i][j] > max):
                    max = data[i][j]
    return max

#takes the files and converts them to usable arrays
def load_and_read_files(filename, string_class_labels):
    data = [None]
    x = [None]
    q = []
    #variable size denotes the dimensionality of the input of the training set
    with open(filename, 'r', encoding='utf-8-sig') as opened_file:
        line = opened_file.readline()
        list = line.split()
        size = len(list)
        opened_file.seek(0, 0)
        for line in opened_file:
            list = line.split()
            for i in range(0, size):
                if(i < (size - 1)):
                    x.append(float(list[i]))
                else:
                    q.append(list[i])
            data.append(x)
            x = [None] 
    (t, s) = convert_class_labels(q, string_class_labels)
    return (data, t, s, size - 1, q)

#takes the string class labels and converts them to one-hot vector  
def convert_class_labels(q, string_class_labels):
    t = [None]
    s = [None]
    if not string_class_labels:
        labels = set(q)
        for label in labels:
            s.append(label)
    else:
        s = string_class_labels
    k = len(s) - 1
    for n in q:
        one_hot_vector = [None]
        for i in range(1, k+1):
            if(n == s[i]):
                one_hot_vector.append(1)
            else:
                one_hot_vector.append(0)
        t.append(one_hot_vector)         
    return (t, s) 
       
neural_network()

##Calculates total runtime for training and testing##
# start = time.time()
# neural_network()
# end = time.time()
# print("Execution time in seconds: ", (end - start))


###Could reduce runtime by combining both randomizers into one