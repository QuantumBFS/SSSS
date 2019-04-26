import torch
import numpy as np
from .template import Target

class Wave(Target):

    def __init__(self):
        super(Wave, self).__init__(2,'Wave')

    def energy(self, x):
        w = torch.sin(np.pi*x[:, 0]/2.)
        return -0.5*((x[:, 1] -w)/0.4)**2