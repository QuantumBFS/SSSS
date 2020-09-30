'''
idea taken from https://math.mit.edu/~stevenj/18.336/adjoint.pdf
'''

import numpy as np 
import torch
torch.set_default_dtype(torch.float64)
import torch.nn as nn

class Schrodinger1D(nn.Module):
    def __init__(self, xmesh):
        super(Schrodinger1D, self).__init__()
        
        self.xmesh = xmesh
        self.potential = nn.Parameter(xmesh**2)

        nmesh = xmesh.shape[0]
        h2 = (xmesh[1]-xmesh[0])**2
        self.K =   torch.diag(1/h2*torch.ones(nmesh,     dtype=xmesh.dtype), diagonal=0) \
                 - torch.diag(0.5/h2*torch.ones(nmesh-1, dtype=xmesh.dtype), diagonal=1) \
                 - torch.diag(0.5/h2*torch.ones(nmesh-1, dtype=xmesh.dtype), diagonal=-1)

    def _solve(self):

        H = torch.diag(self.potential) + self.K
        _, psi = torch.symeig(H, eigenvectors=True) 

        return psi[:, 0] # 0 for ground state

    def forward(self, target):
        psi = self._solve()
        return (psi**2 - target).abs().sum()

    def plot(self, target):
        psi = self._solve()

        plt.cla()
        plt.plot(self.xmesh.numpy(), target.numpy(), label='target')
        plt.plot(self.xmesh.numpy(), psi.square().detach().numpy(), label='current')
        plt.plot(self.xmesh.numpy(), self.potential.detach().numpy()/10000, label='V/10000')
        plt.legend()
        plt.draw()

if __name__=='__main__':
    #prepare mesh and target density
    xmin = -1; xmax = 1; Nmesh = 500
    xmesh = torch.linspace(xmin, xmax, Nmesh)
    
    target = torch.zeros(Nmesh)
    idx = torch.where(torch.abs(xmesh)<0.5)
    target[idx] = 1.-torch.abs(xmesh[idx])
    target = (target/torch.norm(target))**2
    
    model = Schrodinger1D(xmesh)
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=10, tolerance_change = 1E-7, tolerance_grad=1E-7, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        loss = model(target) # density difference 
        loss.backward()
        return loss 

    import matplotlib.pyplot as plt 
    plt.ion()
    for epoch in range(50):
        loss = optimizer.step(closure)
        print (epoch, loss.item())
        model.plot(target)
        plt.pause(0.01)
    plt.ioff()

    model.plot(target)
    plt.show()
