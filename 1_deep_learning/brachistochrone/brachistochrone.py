import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torchdiffeq import odeint_adjoint as odeint
from exact_solver import solver

class MLP(nn.Module):
    def __init__(self, hidden_size, y1=1.0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        self.y1 = y1

    def _f(self, x):
        out = F.softplus(self.fc1(x))
        out = F.softplus(self.fc2(out))
        out = self.fc3(out)
        return out.sum()

    def forward(self, x):
        '''
        y(0) = 0
        y(1) = y1
        '''
        f0 = self._f(torch.tensor([0.0]))
        f1 = self._f(torch.tensor([1.0]))
        return self._f(x) - (f0 + self.y1)*(1.0-x) - f1*x + self.y1

    def value_and_grad(self, x):
        y = self.forward(x)
        return y, torch.autograd.grad(y, x, grad_outputs=torch.ones(x.shape[0]), create_graph=True)[0]

class Brachistochrone(nn.Module):
    def __init__(self, g, v0, net):
        super(Brachistochrone, self).__init__()
        self.v0 = v0
        self.g = g
        self.net = net

    def forward(self, x, t):
        with torch.enable_grad():
            y, dydx = self.net.value_and_grad(x.view(-1).detach().requires_grad_())
        return torch.sqrt((1+dydx**2)/(2*self.g*y+ self.v0**2))

def plot(model,para):
    plt.cla()
    xlist = torch.linspace(0.0, 1.0, 21)
    ylist = [model.net(torch.tensor([x])) for x in xlist]
    plt.plot(xlist.numpy(), ylist, lw=2,label='learned curve')
    plt.plot([0.0, 1.0], [0.0, model.net.y1], 'r*', ms=20)
    plt.gca().invert_yaxis()

    tlist = np.linspace(para[2],para[3],21)
    xlist = para[0]*(tlist- np.sin(tlist)) - para[1]
    ylist = para[0]*(1 - np.cos(tlist)) -para[4]
    plt.plot(xlist,ylist,lw=2,label='exact')
    plt.legend(loc='upper right')

    plt.xlabel('$x$')
    plt.ylabel('$y$')

    plt.draw()
    plt.pause(0.01)

if __name__ == '__main__':

    g = 10.0  #gravity
    v0 = 1.0  #initial velocity
    nh = 32   #number of hidden neurons
    y1 = 1.0  #fininal y coordinate

    para = np.append(solver(v0,g,y1),v0**2/(2*g)) # exact solution as a reference
    tbest = (para[3]-para[2])*np.sqrt(para[0]/g)

    model = Brachistochrone(g, v0, MLP(nh, y1))
    optimizer = optim.Adam(model.parameters(), lr=1E-2)

    import matplotlib.pyplot as plt
    # Set up figure.
    fig = plt.figure(figsize=(8,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)

    for epoch in range(100):
        optimizer.zero_grad()
        t = odeint(model, torch.tensor([0.0]), torch.tensor([0.0, 1.0]))
        loss = t[1] - t[0]
        loss.backward()
        optimizer.step()
        print ("step = %d"%epoch, "time = %.5f"%loss.item(), " (exact = %.5f)"%tbest)
        plot(model,para)
