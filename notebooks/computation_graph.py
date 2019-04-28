from __future__ import division
import numpy as np

class Dense(object):
    '''
    linear node f(x) = xW + b.

    Attributes:
        parameters (list): variables (input nodes) that directly feed into this node, W and b.
        parameters_delta (list): gradients for parameters.
    '''
    def __init__(self, input_shape, output_shape, mean=0, variance=0.01):
        self.params = [mean + variance * np.random.randn(input_shape, output_shape),
                           mean + variance * np.random.randn(output_shape)]
        self.params_delta = [None, None]

    def forward(self, x, *args):
        '''function itself.'''
        self.x = x
        return np.matmul(x, self.params[0]) + self.params[1]

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
    def forward(self, x, *args):
        self.x = x
        self.y = 1.0 / (1.0 + np.exp(-x))
        return self.y

    def backward(self, delta):
        return delta * ((1 - self.y) * self.y)

class Mean(F):
    '''Mean function module'''
    def forward(self, x, *args):
        self.x = x
        return x.mean()

    def backward(self, delta):
        return delta * np.ones(self.x.shape) / np.prod(self.x.shape)

class Net(object):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x 

    def backward(self):
        y_delta = 1.0
        for l in self.layers[::-1]:
            y_delta = l.backward(y_delta)
        return y_delta

if __name__=='__main__':
    np.random.seed(42)
    
    n_batch = 16
    n_in = 10
    n_out = 20
    net = Net([Dense(n_in, n_out), Sigmoid(), Mean()])

    def func(x):
        x = x.reshape(n_batch, n_in)
        return net.forward(x)
    
    def grad(x):
        x = x.reshape(n_batch, n_in)
        net.forward(x)
        return net.backward().reshape(n_batch*n_in)

    from scipy.optimize import check_grad
    x = np.random.randn(n_batch*n_in)
    check_grad(func, grad, x)
