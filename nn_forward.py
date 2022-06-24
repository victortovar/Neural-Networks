import numpy as np
import os
import sys


# (layers, units, biases, weights) = nn_load(pathname)
# pathname: specifies the name of a local file
# layers: an integer, specifies the number of layers in the network.
# units: a list of integers specifying the number of units in each layer.
    # units[L] is the number of units in layer L (where layer 1 is the input layer).
    # units[0] = None, since in our notation there is no layer 0.
    # units[1] is the number of units in the input layer
    # units[layers] is the number of units in the output layer
# biases: a list of numpy column matrices. 
    # biases[L] contains the biases in layer L (where layer 1 is the input layer).
    # biases[0] = None, since in our notation there is no layer 0.
    # biases[1] = 1, since the input layer contains no perceptrons and thus no bias values
    # biases[2] contains the biases in the first hidden layer
    # biases[layers] contains the biases in the output layer.
# weights: a list of numpy matrices specifying the weights in each layer.
    # weights[L] contains the weights in layer L (where layer 1 is the input layer).
        # Its number of rows is the number of units in layer L (saved as units[L])
        # Its number of columns is the number of units in layer L-1 (saved as units[L-1])
        # Every row specifies the weights in a unit of layer L
    # weights[0] = None, since in our notation there is no layer 0.
    # weights[1] = 1, since the input layer contains no perceptrons and thus no weight values
    # weights[2] contains the weights in the first hidden layer
    # weights[layers-2] contains the biases in the output layer.
# With the exception of the input layer, each layer L is assumed to be fully 
# connected, meaning that the input of each unit in layer L is connected
# to the outputs of all units in layer L-1.

def nn_load(pathname):
    if not(os.path.isfile(pathname)):
        print("read_data: %s not found", pathname)
        return None

    in_file = open(pathname)
    
    # get number of layers
    line = in_file.readline()
    items = line.split()
    if (len(items) != 2) or (items[0] != "layers:"):
        print("nn_load: invalid layers line %s" %(line))
        in_file.close()
        return None
    
    layers = int(items[1])

    # get number of units in each layer
    units = (layers+1) * [None]
    line = in_file.readline()
    items = line.split()
    if (len(items) != layers+1) or (items[0] != "units:"):
        print("nn_load: invalid units line %s" %(line))
        in_file.close()
        return None
    
    for i in range(0, layers):
        units[i+1] = int(items[i+1])
    
    # read the biases and weights for each layer
    biases = [None, None]
    weights = [None, None]
    
    for i in range(2, layers+1):
        current_units = units[i]
        previous_units = units[i-1]
        
        line = in_file.readline().strip()
        if (line != "start layer"):
            print("nn_load: expected 'start layer', invalid line %s" %(line))
            in_file.close()
            return None

        # read current bias
        line = in_file.readline().strip()
        if (line != "start bias"):
            print("nn_load: expected 'start bias', invalid line %s" %(line))
            in_file.close()
            return None
        b = read_matrix(in_file, 1, current_units)
        b = b.transpose()
        biases.append(b)
        line = in_file.readline().strip()
        if (line != "end bias"):
            print("nn_load: expected 'end bias', invalid line %s" %(line))
            in_file.close()
            return None
    
        # read current weights matrix
        line = in_file.readline().strip()
        if (line != "start w"):
            print("nn_load: expected 'start w', invalid line %s" %(line))
            in_file.close()
            return None
        w = read_matrix(in_file, current_units, previous_units)
        weights.append(w)
        # read current weights matrix
        line = in_file.readline().strip()
        if (line != "end w"):
            print("nn_load: expected 'start w', invalid line %s" %(line))
            in_file.close()
            return None
        
    in_file.close()
    return (layers, units, biases, weights)

def sigmoid_func(a):
    e = np.exp(-a)
    sig = 1/(1+e)
    return sig

def step_func(x):
    if(x < 0):
        return 0
    else:
        return 1

def read_input_file(filename):
    input = []
    in_file = open(filename)
    for line in in_file.readlines():
        input.append(float(line))
    in_file.close()
    return input



def output(layers, units, biases, weights, input_list, activation):
    # a and z values for each layer will be stored here
    input = read_input_file(input_list)
    a_list = []
    z_list = []
    a_values = []
    z_values = []
    z_values.append(input)
    for i in range(2, len(biases)):
        z_list = []
        a_list = []
        for j in range(0, len(biases[i])):
            
            a = biases[i][j] + weights[i][j].dot(z_values[i-2])
            a_list.append(a[0])
            if(activation == "step"):
                z = step_func(a[0])
            else:
                z = sigmoid_func(a[0])
            z_list.append(z)
        a_values.append(a_list)
        
        z_values.append(z_list)
    print_results(a_values, z_values)

def print_results(a_values, z_values):    
    fa = []
    fa_list = []
    for i in range(len(a_values)):
        for j in range(len(a_values[i])):
            output = float("{0:.4f}".format(a_values[i][j]))
            fa_list.append(output)
        fa.append(fa_list)
        fa_list = []
    
    fz = []
    fz_list = []
    for i in range(len(z_values)):
        for j in range(len(z_values[i])):
            output = float("{0:.4f}".format(z_values[i][j]))
            fz_list.append(output)
        fz.append(fz_list)
        fz_list = []

    
    for i, v in enumerate(fa):
        print("layer %d, a values: " % (i + 1), v)
    print()
    for i, v in enumerate(fz):
        print("layer %d, z values: " % (i), (v))
    print()
    
# arguments:
#    in_file: source file of the data
#    rows, cols: size of the matrix that should be read 
# returns:
#    result: a matrix of size rows x cols
def read_matrix(in_file, rows, cols):
    result = np.zeros((rows, cols))
    
    for row in range(0, rows):
        file_line = in_file.readline()
        items = file_line.split()
        cols2 = len(items)
        if (cols2 != cols):
            print("read_matrix: Line %d, %d columns expected, %d columns found" %(row, cols, cols2))
            return None
        for col in range(0, cols):
            result[row][col] = float(items[col])

    return result
     

def print_nn_info(layers, units, biases, weights):
    print("There are %d layers" % (layers))
    print("Units in each layer:", units[1:])
    for i in range(2, len(biases)):
        print("layer %d biases:" % (i), biases[i][:,0])
    print()
    for i in range(2, len(weights)):
        print("layer %d weights:" % (i))
        print(weights[i])


def test_nn_load():
    default_directory = "."
    default_nn_file = "nn_xor.txt"
    
    if (len(sys.argv) >= 2):
        nn_file = sys.argv[1]
    else:
        nn_file = default_directory + "/" + default_nn_file
    
    (layers, units, biases, weights) = nn_load(nn_file)
    input_list = sys.argv[2]
    activation = sys.argv[3]
    output(layers, units, biases, weights, input_list, activation)

test_nn_load()
