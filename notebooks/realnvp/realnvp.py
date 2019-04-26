import math 
import torch
import torch.nn as nn

class NVPCouplingLayer(nn.Module):

    def __init__(self, map_s, map_t, b):
        super(NVPCouplingLayer , self).__init__()
        self.map_s = map_s
        self.map_t = map_t
        self.b = b.clone().unsqueeze(0)

    def forward(self, x):
        self.logjac = x.new_zeros(x.shape[0])
        s, t = self.map_s(self.b * x), self.map_t(self.b * x) 
        y = self.b * x + (1 - self.b) * (torch.exp(s) * x + t) 
        self.logjac += ((1 - self.b) * s).sum(1)
        return y 

    def inverse(self, y):
        self.logjac = y.new_zeros(y.shape[0])
        s, t = self.map_s(self.b * y), self.map_t(self.b * y)
        self.logjac -= ((1 - self.b) * s).sum(1)
        y = self.b * y + (1 - self.b) * (torch.exp(-s) * (y - t))
        return y 

class NVPNet(nn.Module):
    def __init__(self, dim, hdim, depth, device='cpu'):
        super(NVPNet, self).__init__() 
        self.dim = dim 
        self.device = device
        b = torch.Tensor(dim).to(device)
        self.layers = nn.ModuleList() 
        for d in range(depth):
            if d%2 == 0:
                # Tag half the dimensions
                i = torch.randperm(b.numel()).narrow(0, 0, b.numel() // 2) 
                b.zero_()[i] = 1
            else: 
                b=1-b

            map_s = nn.Sequential(nn.Linear(dim, hdim), 
                                  nn.ELU(), 
                                  nn.Linear(hdim, hdim),
                                  nn.ELU(), 
                                  nn.Linear(hdim, dim)
                                  ) 
            map_t = nn.Sequential(nn.Linear(dim, hdim), 
                                  nn.ELU(), 
                                  nn.Linear(hdim, hdim),
                                  nn.ELU(), 
                                  nn.Linear(hdim, dim)
                                  ) 
            self.layers.append(NVPCouplingLayer(map_s, map_t, b))

    def forward(self, x):
        self.logjac = x.new_zeros(x.shape[0])
        for m in self.layers: 
            x = m(x) 
            self.logjac += m.logjac
        return x

    def inverse(self, y):
        self.logjac = y.new_zeros(y.shape[0])
        for m in reversed(self.layers): 
            y = m.inverse(y) 
            self.logjac += m.logjac
        return y

    def sample(self, batch_size):
        z = torch.Tensor(batch_size, self.dim).normal_() 
        x = self.forward(z)
        logp = - 0.5 * z.pow(2).add(math.log(2 * math.pi)).sum(1)  - self.logjac
        return x, logp 

    def logprob(self, x):
        z = self.inverse(x)
        return - 0.5 * z.pow(2).add(math.log(2 * math.pi)).sum(1)  + self.logjac

    def save(self, save_dict):
        for d, layer in enumerate(self.layers):
            save_dict['map_s'+str(d)] = layer.map_s.state_dict()
            save_dict['map_t'+str(d)] = layer.map_t.state_dict()
            save_dict['mask'+str(d)] =  layer.b.to('cpu')
        return save_dict

    def load(self, save_dict):
        for d, layer in enumerate(self.layers):
            layer.map_s.load_state_dict(save_dict['map_s'+str(d)])
            layer.map_t.load_state_dict(save_dict['map_t'+str(d)])
            layer.b = save_dict['mask'+str(d)].to(self.device)
        return save_dict

if __name__=='__main__':

    batch_size = 100
    dim = 2

    model = NVPNet(dim = dim, hdim = 4, depth = 8)
    z = torch.randn(batch_size, dim, requires_grad=True)

    x = model.forward(z) # generate a new dataset.
    x_logjac = model.logjac # record log(Jacobian) in generate process.
    print (model.logjac)

    z_infer = model.inverse(x) # inference back to the original dataset.
    z_infer_logjac = model.logjac # record log(Jacobian) in inference process.
    print (model.logjac)

    from numpy.testing import assert_array_almost_equal
    assert_array_almost_equal(z_infer.data.numpy(),z.data.numpy()) # test if they are the same.
    assert_array_almost_equal(x_logjac.data.numpy(),-z_infer_logjac.data.numpy()) # abs(log(Jacobian))
