## Multilayer Neural Network from Scratch

This is an exercise to build up a multilayer neural network from scratch using
only the numpy library for binary classification. A number of different
optimization algorithm options are offered to improve the performance of
standard gradient descent.

### How to use

First instantiate a NeuralNet object with the training X and Y data sets as
well as a list of the layers and their dimensions.  Training and validation
data must be formatted so that features are the rows and observations are the
columns.  Since this is only for binary classification the dimension of the
last layer must always be one. For example:

    clf = NeuralNet(X_t, y_t, [6, 3, 1], initialization = 'he', rseed=9)
    clf.train(0.3, 500)

This code will create a neural network object with 6 hidden layers in the first
layer and 3 hidden layers in the second layer.


See the notebook tests_and_examples.ipynb for an example workflow on the MNIST
dataset.


### Initialization and training parameters

Initialization Parameters

**X** : np.array()

A numpy array containing the training x values in the dataset. The rows
of the array should be the features and the columns of the array should
be the observations: shape(n_x, m), where m is the number of
observations.


**y** : np.array()

The corresponding y values from the training data set. This should be a
single row array with one column for each observation in the dataset:
shape(1, m)


**layers** : list()

A list representing the number of layers and the dimension of each
hidden layer in the neural network.  For example [4, 4, 2, 1]
indicates a neural network 4 hidden units in the first layer, 4
hidden units in the second layer, to hidden units in the third layer,
and one unit in the last layer for classification.  The dimension of
the last layer layer must always be one for binary classification.


**initialization** : str()

Specify the type of parameter initially is a initialization for the
weights W.  Specifying 'he' will use he initialization instead of
random.


**r_seed** : int

Specify a random seed for numpy to create reproducable results.


Training Parameters

**lr0** : float

The starting learning rate when training


**num_epochs** : int

The number of iterations on the whole training data


**lambd** : float, optional

Set the strength of L2 regularization applied during training.
Defaults to 0 which is no regularization.


**batch_size** : int, optional

The batch size for minibatch gradient descent. Use 1 for
stochastic gradient descent. Defaults to None which tells the
algorithm to use the entire training set (full batch).

**beta** : float, optional

Set the strength of momentum used in updating the weights.
Defaults to 0 which indicates no momentum and can range from 0
to 1.


**decay_rate** : float, optional

Set the learning rate decay. This should be from 0 to 1 both higher
decay indicating a bigger drop.  This defaults to 0 which indicates
no decay.


**time_interval** : int, optional

Sets how frequently the learning rate decay is applied. The default
value of 100 indicates that the learning rate drops every 100
epochs.


### Mathematical formulas and derivation**

Coming soon