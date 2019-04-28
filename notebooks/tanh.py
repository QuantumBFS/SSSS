'''
Follows HIPS autograd https://github.com/HIPS/autograd/blob/master/examples/tanh.py
'''

import torch
import matplotlib.pyplot as plt

x = torch.linspace(-7, 7, 100, requires_grad=True)

for i in range(7):
    if (i==0):
        y = torch.tanh(x/2)
    else:
        y, = torch.autograd.grad(y, x, grad_outputs=torch.ones(y.shape[0]), create_graph=True)

    plt.plot(x.detach().numpy(), y.detach().numpy(), '-', label='$%g$'%(i))

plt.legend()
plt.show()
