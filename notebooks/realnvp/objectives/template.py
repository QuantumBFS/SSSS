import torch
from torch import nn 

class Target(nn.Module):
    '''
    base class for target 
    '''
    def __init__(self,nvars,name = "Target"):
        super(Target, self).__init__()
        self.nvars = nvars
        self.name = name

    def __call__(self, x):
        return self.energy(x)

    def energy(self,z):
        raise NotImplementedError(str(type(self)))
