import torch
import numpy as np 
from .template import Target

class Mog2(Target):

    def __init__(self, offset=0.8):
        super(Mog2, self).__init__(2,'Mog2')
        self.offset = offset

    def energy(self, x):

        v1 = torch.sqrt((x[:,0]-self.offset)**2 + (x[:, 1]-self.offset)**2)*2.
        v2 = torch.sqrt((x[:,0]+self.offset)**2 + (x[:, 1]+self.offset)**2)*2.

        pdf1 = torch.exp(-0.5* v1*v1) /np.sqrt(2*np.pi * 0.25)
        pdf2 = torch.exp(-0.5* v2*v2) /np.sqrt(2*np.pi * 0.25)

        return torch.log(0.5*pdf1 + 0.5* pdf2)
