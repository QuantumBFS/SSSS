import torch
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

        return psi[:, 0]

    def forward(self, target):
        psi = self._solve()
        return (psi*target).abs().sum()

    def plot(self, target):

        psi = self._solve()

        plt.cla()
        plt.plot(self.xmesh.numpy(), target.numpy(), label='target')
        plt.plot(self.xmesh.numpy(), np.abs(psi.detach().numpy()), label='current')
        plt.plot(self.xmesh.numpy(), self.potential.detach().numpy()/10000, label='V/10000')
        plt.legend()
        plt.draw()

if __name__=='__main__':
    import numpy as np    
    #prepare mesh and target
    xmin = -1; xmax = 1; Nmesh = 300
    xmesh = np.linspace(xmin, xmax, Nmesh)
    
    target = np.zeros(Nmesh)
    idx = np.where(np.abs(xmesh)<0.5)
    target[idx] = 1.-np.abs(xmesh[idx])
    target = target/np.linalg.norm(target)
    
    xmesh = torch.from_numpy(xmesh) 
    target = torch.from_numpy(target)

    model = Schrodinger1D(xmesh)
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=10)

    def closure():
        optimizer.zero_grad()
        loss = 1.-model(target) # infidelity
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
