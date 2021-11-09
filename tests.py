"""Test out on a simple dataset."""


import numpy as np
from NeuralNetNLayer import NeuralNet


A = np.array([[1, -1, -3, -1, 1, 1], 
             [2, 2, 2, 2, 2, 2],
             [2, 0, 1, 0, 3, 3]])

b = np.array([1, 0, 0, 0, 1, 1]).reshape(1, 6)

my_nn = NeuralNet(A, b, [4, 3, 1])

my_nn.train(1, 1000)

print(my_nn.training_accuracy)
