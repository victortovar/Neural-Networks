#Victor Tovar
#1000874189
#Assignment 1 Task 10

import numpy as np


def nth_smallest(data, top, bottom, left, right, n):
    list = []
    for i in range(top, bottom+1):
        for j in range(left, right+1):
            list.append(data[i][j])
    list.sort()
    return list[n-1]
    
a = np.array([[0.8147, 0.0975, 0.1576, 0.1419, 0.6557, 0.7577],
       [0.9058, 0.2785, 0.9706, 0.4212, 0.4212, 0.7431],
       [0.127 , 0.5469, 0.9572, 0.9157, 0.8491, 0.3922],
       [0.9134, 0.9575, 0.4854, 0.7922, 0.934 , 0.6555],
       [0.6324, 0.9649, 0.8003, 0.9595, 0.6787, 0.1712]])
print(nth_smallest(a, 1, 2, 3, 4, 1))
print(nth_smallest(a, 1, 2, 3, 4, 2))
print(nth_smallest(a, 1, 2, 3, 4, 3))
print(nth_smallest(a, 1, 2, 3, 4, 4))

