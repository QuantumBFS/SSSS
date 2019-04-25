import torch
torch.manual_seed(42)
import torch.nn as nn
import numpy as np 

def bget(i,p):
    '''
    return the p-th bit of the word i
    '''
    return (i >> p) & 1

def bflip(i,p):
    '''
    return the integer i with the bit at position p flipped: (1->0, 0->1)
    '''
    return i ^(1<<p)

class TFIM(nn.Module):
    '''
    1D transverse field Ising model 
    '''
    def __init__(self, L):
        super(TFIM, self).__init__()

        self.L = L
        self.Gamma = nn.Parameter(0.01*torch.randn(1))

    def _buildH(self):
        Nstates = 1 << self.L
        H = torch.zeros(Nstates, Nstates)
        # loop over all basis states
        for i in range(Nstates):
            #diagonal term
            for site in range(self.L):
                H[i,i] += -(2*bget(i, site)-1) * (2*bget(i, site+1)-1)
            
            #off-diagonal term
            for site in range(self.L):
                j = bflip(i, site)
                H[i,j] = -self.Gamma
        return H

    def ee(self):
        '''
        bipartite entanglement entropy of ground state 
        '''
        _, v = torch.symeig(self._buildH(), eigenvectors=True)

        #WIP 

        return ... 

if __name__=='__main__':
    import sys 
    L = 4
    model = TFIM(L)

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-1)

    params = list(model.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = sum([np.prod(p.size()) for p in params])
    print ('total nubmer of trainable parameters:', nparams)
    
    for e in range(100):
        loss = -model.ee() 
        
        model.zero_grad()
        loss.backward()
        optimizer.step()
        print (e, loss.item())
