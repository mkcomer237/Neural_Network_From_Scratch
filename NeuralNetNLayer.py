"""Create a neural network class that can be trained with n layers.

See the readme for a more complete description of the training steps.
"""


import numpy as np


class NeuralNet(object):
    """Neural network accepting a list of layers."""

    def __init__(self, X, y, layers=[4, 4, 2, 1], initialization='random',
                 rseed=None):
        """Initialize all of the model parameters.

        Takes in a list of layers and their size. Use dictionaries to store
        W, B, Z, and A parameters.  Also initialize the W and B parameters
        using either random or he initalization and accept a random state.
        These will be global parameters that are overwritten with each training
        batch.

        X and y are also initialized and synchronously randomized here but
        then further subsegmented during mini-batch training.
        """

        # Dictionaries to store each value by layer number
        self.W = {}
        self.B = {}
        self.Z = {}
        self.A = {}
        self.dW = {}
        self.dB = {}
        self.dZ = {}
        self.dA = {}

        self.layers = [np.shape(X)[0]] + layers
        self.n_layers = len(self.layers)
        self.L = self.n_layers - 1

        # Initalize the weight and bias parameters.
        # following hidden layers have both r and c dimensions equal to the
        # n hidden units
        if initialization not in ['random', 'he']:
            raise Exception("Initalization must be either 'random' or 'he'")

        rng = np.random.RandomState(rseed)
        for i in range(1, self.n_layers):
            # print(i, i+1)
            # rows are the number of features in the previous layer, cols are
            # the next layer
            if initialization == 'he':
                self.W[i] = (rng.randn(self.layers[i], self.layers[i-1])
                             * np.sqrt(2/self.layers[i-1]))
                self.B[i] = np.zeros((self.layers[i], 1))
            if initialization == 'random':
                self.W[i] = (rng.randn(self.layers[i], self.layers[i-1])
                             * 0.01)
                self.B[i] = np.zeros((self.layers[i], 1))

        # Set other base parameters
        # Shuffle the training data to prepare for minibatch
        self.m = np.shape(X)[1]
        perm = np.random.permutation(self.m)
        self.X = X[:, perm]
        self.y = y[:, perm]

    def relu(self, z):
        """Implement the relu activation function."""
        return np.maximum(0, z)

    def relu_derivative(self, z):
        """Implement the derivative of the relu function."""
        return np.where(z < 0, 0, 1)

    def sigmoid_np(self, z):
        """Implement the sigmoid function."""
        return 1/(1+np.exp(-1*z))

    def forward_prop(self, X):
        """Forward propagation step.

        The global A[l] is overwritten here regardless of what X is being
        used to initiate forward prop."""

        self.A[0] = X

        for i in range(1, self.n_layers):
            # print(i)
            # W transpose not neccessary because of how we defined the matrix
            self.Z[i] = np.dot(self.W[i], self.A[i-1]) + self.B[i]
            if i == self.n_layers-1:
                # The last layer needs a sigmoid activation function
                self.A[i] = self.sigmoid_np(self.Z[i])
            else:
                self.A[i] = self.relu(self.Z[i])

    def cost_function(self, y, m, lambd):
        """Calculate the cost function J.

        Optional regularization depending on lambda, defaults to no
        regularization (lambd=0).
        """

        l2_reg_term = 0

        for i in range(1, self.n_layers):
            frobenius_norm = np.sum(np.square(self.W[i]))
            l2_reg_term += (lambd/(2*m))*frobenius_norm

        self.J = (-1 * (1 / m)*(np.dot(y, np.log(self.A[self.L].T)) +
                                np.dot(1-y, np.log(1-self.A[self.L].T)))
                  + l2_reg_term)

        # print(self.J.ravel()[0])

    def backward_prop(self, y, m, lambd):
        """Backward propagation step.

        Regularization is also implemented here depending on lambda.
        lambd = 0 (the default value) means no regularization.
        """

        # Initialize the last layer with the sigmoid gradient
        self.dZ[self.L] = -y + self.A[self.L]
        self.dW[self.L] = ((1 / m)*np.dot(self.dZ[self.L],
                                          self.A[self.L-1].T) +
                           (lambd / m) * self.W[self.L])
        self.dB[self.L] = (1 / m)*np.sum(self.dZ[self.L], axis=1,
                                         keepdims=True)
        # this is used in the next layer's DZ
        self.dA[self.L-1] = np.dot(self.W[self.L].T, self.dZ[self.L])

        # Calculate the gradients for the rest of the layers
        for i in reversed(range(1, self.n_layers-1)):
            # print(i)
            # dA[i] comes from the previous backprobagation step - this is key
            self.dZ[i] = self.dA[i] * self.relu_derivative(self.Z[i])
            self.dW[i] = ((1 / m)*np.dot(self.dZ[i], self.A[i-1].T) +
                          (lambd / m) * self.W[i])
            self.dB[i] = (1 / m)*np.sum(self.dZ[i], axis=1, keepdims=True)
            self.dA[i-1] = np.dot(self.W[i].T, self.dZ[i])

    def train(self, lr, num_iterations, lambd=0, batch_size=None, beta=0):
        """Use gradient descent to train the model.

        The lambd parameter implements L2 regularization, and defaults to
        lambda=0, which is no regularization.

        the batch size parameter controls the training batch size. For example:
        batch_size=1 is stochastic gradient descent.
        batch_size=256 is minibatch gradient descent with batch size of 256.
        batch_size=m (default) is batch gradient descent on all training
        examples at once.

        Momentum is used to update the weights if the parameter beta is greater
        than 0. The v parameter the is the velocity from the previous steps
        dw."""

        # initialize the velocity parameter

        v_W = {}
        v_B = {}
        for i in range(1, self.n_layers):
            v_W[i] = np.zeros(self.W[i].shape)
            v_B[i] = np.zeros(self.B[i].shape)

        # check batch size
        if batch_size is None:
            batch_size = self.m
        elif batch_size > self.m:
            raise Exception('Batch size is too large')

        print('batch size: ', batch_size)

        for epoch in range(num_iterations):

            # Mini batch inner loop
            for i in np.arange(0, self.m, batch_size):

                # subset the training data according to the batch size
                end = min(i + batch_size, self.m)
                X = self.X[:, i:end]
                y = self.y[:, i:end]
                m = end-i

                # print('X:', X, y)
                # Forward and backward propagation
                self.forward_prop(X)
                self.cost_function(y, m, lambd)
                self.backward_prop(y, m, lambd)

                # Update weights with the gradients using momentum
                # if beta == 0 this is equivalent to not using momentum
                for ly in range(1, self.n_layers):
                    v_W[ly] = beta * v_W[ly] + (1 - beta) * (self.dW[ly])
                    v_B[ly] = beta * v_B[ly] + (1 - beta) * (self.dB[ly])
                    self.W[ly] -= lr * v_W[ly]
                    self.B[ly] -= lr * v_B[ly]

            # Print training accuracy
            # This is measured on the entire dataset
            if epoch % 100 == 0:
                print('Iteration:', epoch, ', Cost, Accuracy',
                      self.J.ravel()[0],
                      self.training_accuracy)
            if np.isnan(self.J):
                print('Y:', self.y)
                print('AL', self.A[self.L])
                break

    @property
    def training_accuracy(self):
        """Calculate accuracy on the whole training dataset.

        Uses a 0.5 probability threshold for classification."""
        self.forward_prop(self.X)  # Calculate A2 (output layer) with latest
        # compare AL and y for accuracy
        self.tp = np.where(self.A[self.L] >= 0.5, 1, 0)
        return float((np.dot(self.y, self.tp.T) + np.dot(1-self.y,
                                                         1-self.tp.T)) /
                     float(self.y.size)*100)

    def validation_accuracy(self, X_v, y_v):
        """Calculate accurace on the validation dataset."""
        # Do forward propagation on the validation set
        Z_v = {}
        A_v = {}
        A_v[0] = X_v

        for i in range(1, self.n_layers):
            # W transpose not neccessary because of how we defined the matrix
            Z_v[i] = np.dot(self.W[i], A_v[i-1]) + self.B[i]
            if i == self.n_layers-1:
                A_v[i] = self.sigmoid_np(Z_v[i])
            else:
                A_v[i] = self.relu(Z_v[i])

        # Calculate and return accuracy
        tp = np.where(A_v[self.L] >= 0.5, 1, 0)
        return float((np.dot(y_v, tp.T) + np.dot(1-y_v, 1-tp.T)) /
                     float(y_v.size)*100)
