import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torchdiffeq import odeint_adjoint as odeint

class MLP(nn.Module):
    def __init__(self, dim, hidden_size, device='cpu', name=None):
        super(MLP, self).__init__()
        self.device = device
        if name is None:
            self.name = 'MLP'
        else:
            self.name = name

        self.fc1 = nn.Linear(dim+1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, t, x):
        x = torch.cat([t.view(-1), x])
        out = F.softplus(self.fc1(x))
        out = F.softplus(self.fc2(out))
        out = self.fc3(out)
        return out.sum()

    def grad(self, t, x):
        return torch.autograd.grad(self.forward(t, x), x, grad_outputs=torch.ones(x.shape[0]), create_graph=True)[0]

class Hamiltonian(nn.Module):
    def __init__(self, net):
        super(Hamiltonian, self).__init__()
        self.net = net
        self.J =  torch.tensor([[0.0, 1.0], [-1.0, 0.0]])

    def forward(self, t, x):
        with torch.enable_grad():
            g = self.net.grad(t, x.detach().requires_grad_())
        return g@self.J

def visualize(model, y_traj, y_pred):
    plt.cla()
    plt.plot(y_pred.detach().numpy()[:, 0], y_pred.detach().numpy()[:, 1], '-o', color='r')
    plt.plot(y_traj.detach().numpy()[:, 0], y_traj.detach().numpy()[:, 1], '-*', color='b')

    dydt = []
    for y in np.linspace(-2, 2, 21):
        for x in np.linspace(-2, 2, 21):
            v =  model(torch.tensor(0.0), torch.tensor([x, y]))
            dydt.append(v/v.norm())
    dydt = torch.cat(dydt).view(21, 21, 2)
    dydt = dydt.detach().numpy()
    y, x = np.mgrid[-2:2:21j, -2:2:21j]
    plt.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="gray")

    plt.gca().set_yticks([])
    plt.gca().set_xticks([])

    plt.xlim([-2, 2])
    plt.ylim([-2, 2])

    plt.draw()
    plt.pause(0.01)

if __name__ == '__main__':

    Nt = 10
    t = torch.linspace(0, 1, Nt)
    y_traj = torch.stack([t*torch.cos(t*np.pi*2), t*torch.sin(t*np.pi*2)], dim=1)

    model = Hamiltonian(MLP(2, 50))
    optimizer = optim.Rprop(model.parameters())

    import matplotlib.pyplot as plt
    # Set up figure.
    fig = plt.figure(figsize=(8,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)

    for epoch in range(200):
        optimizer.zero_grad()
        y_pred = odeint(model, y_traj[0], t)
        loss = ((y_traj - y_pred)**2).mean()
        loss.backward()
        optimizer.step()

        print (epoch, loss.item())
        visualize(model, y_traj, y_pred)
    plt.ioff()

    visualize(model, y_traj, y_pred)
    plt.show()
