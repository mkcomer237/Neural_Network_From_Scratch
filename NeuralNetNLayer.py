"""Create a neural network class that can be trained with n layers.

See the readme for a more complete description of the training steps.
"""


import numpy as np


class NeuralNet(object):
    """Neural network accepting a list of layers."""

    def __init__(self, X, y, layers=[4, 4, 2, 1]):
        """Initialize all of the model parameters.
        
        Takes in a list of layers and their size. Use dictionaries to store 
        W, B, Z, and A parameters.
        """
        # dictionaries to store each value by layer number
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
        
        #following hidden layers have both r and c dimensions equal to the n hidden units 
        for i in range(1, self.n_layers):
            #print(i, i+1)
            self.W[i] = np.random.randn(self.layers[i], self.layers[i-1])*0.01 # rows are the number of features in the previous layer, cols are the next layer
            self.B[i] = np.zeros((self.layers[i], 1))

        # Set other key parameters 
        self.m = np.shape(X)[1]
        self.X = X
        self.A[0] = X
        self.y = y     
    
    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return np.where(z < 0, 0, 1)
    
    def sigmoid_np(self, z):
        return 1/(1+np.exp(-1*z))
    
    def forward_prop(self):
        
        for i in range(1, self.n_layers):
            #print(i)
            self.Z[i] = np.dot(self.W[i], self.A[i-1]) + self.B[i] # W transpose not neccessary because of how we defined the matrix
            if i == self.n_layers-1:
                self.A[i] = self.sigmoid_np(self.Z[i]) 
            else:
                self.A[i] = self.relu(self.Z[i]) 
        
        # The last layer needs a sigmoid activation function 
            
    def cost_function(self):
        self.J = -1*(1/self.m)*(np.dot(self.y, np.log(self.A[self.L].T))+
                                np.dot(1-self.y, np.log(1-self.A[self.L].T)))
        #print(self.J.ravel()[0])
        
    def backward_prop(self):
        
        # Initialize the last layer with the sigmoid gradient 
        self.dZ[self.L] = -self.y + self.A[self.L] 
        self.dW[self.L] = (1/self.m)*np.dot(self.dZ[self.L], self.A[self.L-1].T)
        self.dB[self.L] = (1/self.m)*np.sum(self.dZ[self.L], axis=1, keepdims=True)
        self.dA[self.L-1] = np.dot(self.W[self.L].T, self.dZ[self.L]) # this is used in the next layer's DZ

        # Calculate the gradients for the rest of the layers
        for i in reversed(range(1, self.n_layers-1)):
            #print(i)
            self.dZ[i] = self.dA[i] * self.relu_derivative(self.Z[i]) # dA[i] comes from the previous backprobagation step - this is the key
            self.dW[i] = (1/self.m)*np.dot(self.dZ[i], self.A[i-1].T)
            self.dB[i] = (1/self.m)*np.sum(self.dZ[i], axis=1, keepdims=True)
            self.dA[i-1] = np.dot(self.W[i].T, self.dZ[i])

        
    def train(self, lr, num_iterations):
        for i in range(num_iterations):
            #execute propagation steps
            self.forward_prop()
            self.cost_function()
            self.backward_prop()
            #print(self.J.ravel()[0])
            if i%100==0:
                print('Iteration:', i, ', Cost, Accuracy', self.J.ravel()[0], self.training_accuracy())
                if np.isnan(self.J):
                    print('Y:', self.y)
                    print('AL', self.A[self.L])
                    break
                #print(self.W[4])
                #print(self.dW[4])
            #Update weights with the gradients 
            #print('W2 before', self.W2)
            #print('updated weights',  self.W2-lr*self.dW2 )
            for i in range(1, self.n_layers):
                self.W[i] -= lr*self.dW[i]
                self.B[i] -= lr*self.dB[i]
            #print('W2 after', self.W2)

    def training_accuracy(self): 
        self.forward_prop() # Calculate A2 (output layer) with the latest weights
        #compare AL and y for accuracy 
        self.tp = np.where(self.A[self.L]>=0.5, 1, 0)
        return float((np.dot(self.y,self.tp.T) + np.dot(1-self.y,1-self.tp.T))/float(self.y.size)*100)

    def validation_accuracy(self, X_v, y_v):
        
        # Do forward propagation on the validation set 
        Z_v = {}
        A_v = {}
        A_v[0] = X_v
        
        for i in range(1, self.n_layers):
            Z_v[i] = np.dot(self.W[i], A_v[i-1]) + self.B[i] # W transpose not neccessary because of how we defined the matrix
            if i == self.n_layers-1:
                A_v[i] = self.sigmoid_np(Z_v[i]) 
            else:
                A_v[i] = self.relu(Z_v[i]) 
        
        #Calculate and return accuracy 
        tp = np.where(A_v[self.L]>=0.5, 1, 0)
        return float((np.dot(y_v, tp.T) + np.dot(1-y_v, 1-tp.T))/float(y_v.size)*100)

        
         