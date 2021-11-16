"""Test out on a simple dataset."""


import numpy as np
from NeuralNetNLayer import NeuralNet


A = np.array([[1, -1, -3, -1, 1, 1, 3],
             [2, 2, 2, 2, 2, 2, 2],
             [2, 0, 1, 0, 3, 3, -1]])

b = np.array([1, 0, 0, 0, 1, 1, 0]).reshape(1, 7)

my_nn = NeuralNet(A, b, [4, 3, 1], initialization='he', rseed=10)

my_nn.train(lr=0.8, num_iterations=10, lambd=0.5, batch_size=4)
print(my_nn.dW[1])
print(my_nn.A[1])
print(my_nn.training_accuracy)
