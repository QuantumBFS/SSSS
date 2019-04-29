import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

from torchdiffeq import odeint

class MLP(nn.Module):
    def __init__(self, dim, hidden_size, device='cpu', name=None):
        super(MLP, self).__init__()
        self.device = device
        if name is None:
            self.name = 'MLP'
        else:
            self.name = name

        self.dim = dim
        self.fc1 = nn.Linear(dim, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1, bias=False)
        self.activation = F.softplus

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = F.softplus(self.fc2(out))
        out = self.fc3(out)
        return out.sum()

    def grad(self, x):
        batch_size = x.shape[0]
        return torch.autograd.grad(self.forward(x), x, grad_outputs=torch.ones(batch_size, device=x.device), create_graph=True)[0]

class Hamiltonian(nn.Module):
    def __init__(self, net):
        super(Hamiltonian, self).__init__()
        self.net = net
        self.J =  torch.tensor([[0.0, 1.0], [-1.0, 0.0]])

    def forward(self, t, y):
        return self.net.grad(y)@self.J

if __name__ == '__main__':
    
    y0 = torch.tensor([[0.0, 0.0]])
    yT = torch.tensor([[1.0, 1.0]]) 
    T = torch.tensor([1.0, 0.0]) 

    model = Hamiltonian(MLP(2, 10))
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    for itr in range(100):
        optimizer.zero_grad()

        y_pred = odeint(model, y0, 1.0)
        loss = (y_pred - yT)**2 

        loss.backward()
        optimizer.step()

        print (itr, loss.item())


