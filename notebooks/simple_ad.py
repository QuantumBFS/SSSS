import numpy as np

class NodeBase(object):

    def __init__(self, data):
        super(NodeBase, self).__init__()
        self._data = data

    def backward(self, delta):
        raise NotImplementedError

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def __repr__(self):
        return 'tracked ' + repr(self.data)

    def __radd__(self, rhs):
        return Node(Add, self, rhs)

    def __ladd__(self, lhs):
        return Node(Add, lhs, self)


class Variable(NodeBase):

    def __init__(self, x, grad=None):
        super(Variable, self).__init__(x)
        self.grad = grad        

    def backward(self, delta):
        if self.grad is None:
            self.grad = delta
        else:
            self.grad += delta
        return None


class Node(NodeBase):

    def __init__(self, f : Functional, *args):
        data = [each.data if isinstance(each, NodeBase) else each for each in args]
        super(Node, self).__init__(f.eval(*data))
        self.args = args
        self.args_data = data
        self.f = f

    def backward(self, delta):
        grads = self.f.gradient(delta, self.data, *self.args_data)
        for each, grad in zip(self.args, grads):
            if isinstance(each, NodeBase):
                each.backward(grad)
        return


class Funtional:

    def eval(self, *args):
        raise NotImplementedError

    def gradient(self, delta, output, *args):
        raise NotImplementedError


class MatMul(Functional):

    @staticmethod
    def eval(A, B):
        return np.matmul(A, B)

    @staticmethod
    def gradient(delta, output, A, B):

        def adjoint(x):
            return np.conj(x.T)

        return np.matmul(delta, adjoint(B)), np.matmul(adjoint(A), delta)


class Add(Functional):

    @staticmethod
    def eval(A, B):
        return A + B

    @staticmethod
    def gradient(delta, output, A, B):
        return delta, delta


class Sigmoid(Functional):

    @staticmethod
    def eval(X):
        return 1. / (1. + np.exp(X))

    @staticmethod
    def gradient(delta, output, X):
        return delta * (1 - output) * output, 


def matmul(A, B):
    return Node(MatMul, A, B)

def sigmoid(X):
    return Node(Sigmoid, X)


if __name__ == '__main__':
    A = Variable(np.random.rand(2, 3))
    B = Variable(np.random.rand(3, 4))
    C = Variable(np.random.rand(2, 4))

    # Dense
    Z = sigmoid(matmul(A, B) + C)
    Z.backward(np.random.rand(2, 4))
    A.grad
