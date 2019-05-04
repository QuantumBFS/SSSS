from __future__ import division
import numpy as np

class Dense(object):
    '''
    linear node f(x) = xW + b.

    Attributes:
        params (list): variables (input nodes) that directly feed into this node, W and b.
        params_delta (list): gradients for parameters.
    '''
    def __init__(self, input_shape, output_shape, mean=0, variance=0.01):
        self.params = [mean + variance * np.random.randn(input_shape, output_shape),
                       mean + variance * np.random.randn(output_shape)]
        self.params_delta = [None, None]

    def forward(self, x, *args):
        '''function itself.'''
        self.x = x # store for backward
        W, b = self.params
        return np.dot(x, W) + b 

    def backward(self, delta):
        '''
        Args:
            delta (ndarray): gradient of L with repect to node's output, dL/dy.

        Returns:
            ndarray: gradient of L with respect to node's input, dL/dx
        '''
        self.params_delta[0] = np.dot(self.x.T, delta)
        self.params_delta[1] = np.sum(delta, 0)
        return np.dot(delta, self.params[0].T)

class F(object):
    '''base class for functions with no parameters.'''
    def __init__(self):
        self.params = []
        self.params_delta = []

class Sigmoid(F):
    '''Sigmoid activation function module'''
    def forward(self, x):
        self.y = 1.0 / (1.0 + np.exp(-x))
        return self.y

    def backward(self, delta):
        return delta * ((1 - self.y) * self.y)

class MSE(F):
    '''Mean function module'''
    def __init__(self, y):
        super(MSE, self).__init__()
        self.y = y

    def forward(self, x):
        self.x = x
        return ((x-self.y)**2).mean()

    def backward(self, delta):
        return delta*2*(self.x-self.y)/np.prod(self.x.shape)

class Sequential(object):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x 

    def backward(self):
        delta = 1.0
        for l in self.layers[::-1]:
            delta = l.backward(delta)
        return delta

if __name__=='__main__':
    np.random.seed(42)
    
    n_batch = 32
    n_in = 1
    n_hidden = 100

    x = np.random.rand(n_batch, n_in)     
    y = (x**2).sum(axis=1, keepdims=True) # size = (n_batch, 1)

    model = Sequential([Dense(n_in, n_hidden), Sigmoid(), Dense(n_hidden, 1), MSE(y)])

    def func(x):
        x = x.reshape(n_batch, n_in)
        return model.forward(x)
    
    def grad(x):
        x = x.reshape(n_batch, n_in)
        model.forward(x)
        return model.backward().reshape(n_batch*n_in)

    from scipy.optimize import check_grad
    x = np.random.randn(n_batch*n_in)
    print ('gradient check:',  check_grad(func, grad, x) ) 
