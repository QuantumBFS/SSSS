import torch
import numpy as np
from .template import Target

class Ring2D(Target):

    def __init__(self):
        super(Ring2D, self).__init__(2,'Ring2D')

    def energy(self, x):
        return -(torch.sqrt((x**2).sum(dim=1))-2.0)**2/0.32

class Ring5(Target):

    def __init__(self):
        super(Ring5, self).__init__(2,'Ring5')

    def energy(self, x):
        x2 = torch.sqrt((x**2).sum(dim=1))
        u1 = (x2 - 1.) **2 /0.04
        u2 = (x2 - 2.) **2 /0.04
        u3 = (x2 - 3.) **2 /0.04
        u4 = (x2 - 4.) **2 /0.04
        u5 = (x2 - 5.) **2 /0.04

        u1 = u1.view(-1, 1)
        u2 = u2.view(-1, 1)
        u3 = u3.view(-1, 1)
        u4 = u4.view(-1, 1)
        u5 = u5.view(-1, 1)

        u = torch.cat((u1, u2, u3, u4, u5), dim=1)
        return -torch.min(u, dim=1)[0]