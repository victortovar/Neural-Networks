import numpy as np
import os
import sys



weights_list = open(sys.argv[1], "r").read().splitlines()
input_list = open(sys.argv[2], "r").read().splitlines()
activation = sys.argv[3]
bias_weight = float(weights_list[0])
weights_list.pop(0)

def sigmoid_func(a):
    e = np.exp(-a)
    sig = 1/(1+e)
    return sig

def step_func(x):
    if(x < 0):
        return 0
    else:
        return 1
    
weights = np.array(weights_list, dtype = float)
input = np.array(input_list, dtype = float)

a = bias_weight + weights.dot(input)

if(activation == "step"):
    if(a < 0):
        z = 0
    else:
        z = 1
else:
    z = sigmoid_func(a)

    
print("a = %.4f\nz = %.4f" % (a, z))