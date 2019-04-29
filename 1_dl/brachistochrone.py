import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torchdiffeq import odeint_adjoint as odeint

class MLP(nn.Module):
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def _f(self, x):
        out = F.softplus(self.fc1(x))
        out = F.softplus(self.fc2(out))
        out = self.fc3(out)
        return out.sum()

    def forward(self, x):
        f0 = self._f(torch.tensor([0.0]))
        f1 = self._f(torch.tensor([1.0]))
        return self._f(x) - (f0-1.0)*(1.0-x) - f1*x + 1.0

    def grad(self, x):
        return torch.autograd.grad(self.forward(x), x, grad_outputs=torch.ones(x.shape[0]), create_graph=True)[0]


class Brachistochrone(nn.Module):
    def __init__(self, net):
        super(Brachistochrone, self).__init__()
        self.net = net

    def forward(self, x, t):
        with torch.enable_grad():
            y = self.net(x.view(-1))
            dydx = self.net.grad(x.view(-1).detach().requires_grad_())
        return torch.sqrt((1+dydx**2)/(2*0.1*y))

def plot(model):
    plt.cla()
    xlist = torch.linspace(0.0, 1.0, 11)
    ylist = [model.net(torch.tensor([x])) for x in xlist]
    plt.plot(xlist.numpy(), ylist)

    plt.draw()
    plt.pause(0.01)

if __name__ == '__main__':

    model = Brachistochrone(MLP(10))
    optimizer = optim.Rprop(model.parameters())

    import matplotlib.pyplot as plt
    # Set up figure.
    fig = plt.figure(figsize=(8,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)

    for epoch in range(100):
        optimizer.zero_grad()
        t = odeint(model, torch.tensor([0.0]), torch.tensor([0.0, 1.0]))
        loss = t[1]
        loss.backward()
        optimizer.step()
        print (epoch, loss.item())
        plot(model)
