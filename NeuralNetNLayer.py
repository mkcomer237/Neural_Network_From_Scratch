"""A neural network class that can be trained with a variable number of layers.

Users can specify the number in dimensions of each later well as a number of
parameters that will inference initialization and the optimization algorithm in
training.
"""


import numpy as np


class NeuralNet(object):
    """Neural network accepting a list of layers.

    To use instantiate the class with the parameters listed below, and call the
    train method to train the model weights.

    Initialization Parameters
    ----------
    X : np.array()
        A numpy array containing the training x values in the dataset. The rows
        of the array should be the features and the columns of the array should
        be the observations: shape(n_x, m), where m is the number of
        observations.
    y : np.array()
        The corresponding y values from the training data set. This should be a
        single row array with one column for each observation in the dataset:
        shape(1, m)
    layers : list()
        A list representing the number of layers and the dimension of each
        hidden layer in the neural network.  For example [4, 4, 2, 1]
        indicates a neural network 4 hidden units in the first layer, 4
        hidden units in the second layer, to hidden units in the third layer,
        and one unit in the last layer for classification.  The dimension of
        the last layer layer must always be one for binary classification.
    initialization : str()
        Specify the type of parameter initially is a initialization for the
        weights W.  Specifying 'he' will use he initialization instead of
        random.
    r_seed : int
        Specify a random seed for numpy to create reproducable results.
    """

    def __init__(self, X, y, layers=[4, 4, 2, 1], initialization='random',
                 rseed=None):
        """Initialize all of the model parameters.

        Take in all of the model parameters specified in the class docstring.

        Use dictionaries to store W, B, Z, and A parameters.  Also initialize
        the W and B parameters using either random or he initalization and
        accept a random state. These will be global parameters that are
        overwritten with each training batch.

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

        # Initalize the weight and bias parameters with random values.
        if initialization not in ['random', 'he']:
            raise Exception("Initalization must be either 'random' or 'he'")

        rng = np.random.RandomState(rseed)
        for i in range(1, self.n_layers):
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
        """
        Forward propagation step.

        Saves the intermediate A[i] and Z[i] for later use, as well as the
        final A value to use in prediction and measuring accuracy.

        Parameters
        ----------
        X : np.array()
            The X training values with features as rows and observations as
            columns.
        """
        self.A[0] = X

        for i in range(1, self.n_layers):
            # W transpose not neccessary because of how we defined the matrix
            self.Z[i] = np.dot(self.W[i], self.A[i-1]) + self.B[i]
            if i == self.n_layers-1:
                # The last layer needs a sigmoid activation function
                self.A[i] = self.sigmoid_np(self.Z[i])
            else:
                self.A[i] = self.relu(self.Z[i])

    def cost_function(self, y, m, lambd):
        """Calculate the cost function J.

        Parameters
        ----------
        y : np.array()
            The true values from the training data.
        m : int
            The number of observations in the training data.
        lambd : float
            be l2 regularization parameter the default of 0 is equivalent
            to no regularization.
        """
        l2_reg_term = 0

        for i in range(1, self.n_layers):
            frobenius_norm = np.sum(np.square(self.W[i]))
            l2_reg_term += (lambd/(2*m))*frobenius_norm

        self.J = (-1 * (1 / m)*(np.dot(y, np.log(self.A[self.L].T)) +
                                np.dot(1-y, np.log(1-self.A[self.L].T)))
                  + l2_reg_term)

    def backward_prop(self, y, m, lambd):
        """Backward propagation step.

        Parameters
        ----------
        y : np.array()
            The true values from the training data.
        m : int
            The number of observations in the training data.
        lambd : float
            be l2 regularization parameter the default of 0 is equivalent
            to no regularization.
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
            # dA[i] comes from the previous backprobagation step - this is key
            self.dZ[i] = self.dA[i] * self.relu_derivative(self.Z[i])
            self.dW[i] = ((1 / m)*np.dot(self.dZ[i], self.A[i-1].T) +
                          (lambd / m) * self.W[i])
            self.dB[i] = (1 / m)*np.sum(self.dZ[i], axis=1, keepdims=True)
            self.dA[i-1] = np.dot(self.W[i].T, self.dZ[i])

    def update_lr(self, lr, epoch, decay_rate, time_interval):
        """Update the learning rate using exponential weight decay."""
        return lr * (1 / (1 + np.floor(epoch / time_interval) * decay_rate))

    def train(self, lr0, num_epochs, lambd=0, batch_size=None, beta=0,
              decay_rate=0, time_interval=100):
        """Use a variation of gradient descent to train the model.

        Parameters
        ----------
        lr0 : float
            The starting learning rate when training
        num_epochs : int
            The number of iterations on the whole training data
        lambd : float, optional
            Set the strength of L2 regularization applied during training.
            Defaults to 0 which is no regularization.
        batch_size : int, optional
            The batch size for minibatch gradient descent. Use 1 for
            stochastic gradient descent. Defaults to None which tells the
            algorithm to use the entire training set (full batch).
        beta : float, optional
            Set the strength of momentum used in updating the weights.
            Defaults to 0 which indicates no momentum and can range from 0
            to 1.
        decay_rate : float, optional
            Set the learning rate decay. This should be from 0 to 1 both higher
            decay indicating a bigger drop.  This defaults to 0 which indicates
            no decay.
        time_interval : int, optional
            Sets how frequently the learning rate decay is applied. The default
            value of 100 indicates that the learning rate drops every 100
            epochs.
        """
        # initialize the velocity parameter

        v_W = {}
        v_B = {}
        for i in range(1, self.n_layers):
            v_W[i] = np.zeros(self.W[i].shape)
            v_B[i] = np.zeros(self.B[i].shape)

        # initialize the learning rate
        lr = lr0

        # check batch size
        if batch_size is None:
            batch_size = self.m
        elif batch_size > self.m:
            raise Exception('Batch size is too large')

        print('batch size: ', batch_size)

        for epoch in range(num_epochs):

            # Mini batch inner loop
            for i in np.arange(0, self.m, batch_size):

                # subset the training data according to the batch size
                end = min(i + batch_size, self.m)
                X = self.X[:, i:end]
                y = self.y[:, i:end]
                m = end-i

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
                print('Iteration:', epoch, ', Cost, Accuracy, lr: ',
                      self.J.ravel()[0],
                      self.training_accuracy,
                      lr)
            if np.isnan(self.J):
                print('Y:', self.y)
                print('AL', self.A[self.L])
                break

            # Update the learning rate using exponential weight decay
            if decay_rate > 0:
                lr = self.update_lr(lr0, epoch, decay_rate, time_interval)

    @property
    def training_accuracy(self):
        """Calculate accuracy on the whole training dataset.

        Uses a 0.5 probability threshold for classification.
        """
        self.forward_prop(self.X)  # Calculate A2 (output layer) with latest
        # compare AL and y for accuracy
        self.tp = np.where(self.A[self.L] >= 0.5, 1, 0)
        return float((np.dot(self.y, self.tp.T) + np.dot(1-self.y,
                                                         1-self.tp.T)) /
                     float(self.y.size)*100)

    def validation_accuracy(self, X_v, y_v):
        """Calculate accurace on the validation dataset.

        Uses a 0.5 probability threshold for classification.

        Parameters
        ----------
        X_v : np.array()
            A numpy array of the X validation set with features as rows and
            observations as columns.
        y_v : np.array()
            A numpy array of the y validation set with one column per
            observation.
        """
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
