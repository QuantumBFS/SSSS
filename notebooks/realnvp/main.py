import torch 
import torch.nn as nn

import numpy as np 
import matplotlib.pyplot as plt 

from realnvp import NVPNet
import objectives

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-cuda", type=int, default=-1, help="use GPU")
    parser.add_argument("-target", default='Ring2D', 
                        choices=['Ring2D', 'Ring5', 'Wave', 'Gaussian', 'Mog2'], help="target distribution")
    parser.add_argument("-batchsize", type=int, default=1024, help="batchsize")
    args = parser.parse_args()
    device = torch.device("cpu" if args.cuda<0 else "cuda:"+str(args.cuda))

    xlimits=[-4, 4]
    ylimits=[-4, 4]
    numticks=31
    x = np.linspace(*xlimits, num=numticks, dtype=np.float32)
    y = np.linspace(*ylimits, num=numticks, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    xy = np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T
    xy = torch.from_numpy(xy).contiguous().to(device)

    # Set up plotting code
    def plot_isocontours(ax, func, alpha=1.0):
        zs = np.exp(func(xy).cpu().detach().numpy())
        Z = zs.reshape(X.shape)
        plt.contour(X, Y, Z, alpha=alpha)
        ax.set_yticks([])
        ax.set_xticks([])
        plt.xlim(xlimits)
        plt.ylim(ylimits)

    target = getattr(objectives, args.target)()
    target.to(device)

    # Set up figure.
    fig = plt.figure(figsize=(8,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)

    model = NVPNet(dim = 2, hdim = 10, depth = 8)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)

    params = list(model.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = sum([np.prod(p.size()) for p in params])
    print ('total nubmer of trainable parameters:', nparams)
    
    np_losses = []
    for e in range(200):
        x, logp = model.sample(args.batchsize)
        loss = logp.mean() - target(x).mean() 
        
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            print (e, loss.item())
            np_losses.append([loss.item()])

            plt.cla()
            plot_isocontours(ax, target, alpha=0.5)
            plot_isocontours(ax, model.logprob)

            samples = x.cpu().detach().numpy()
            plt.plot(samples[:, 0], samples[:,1],'o', alpha=0.8)

            plt.draw()
            plt.pause(0.01)

    np_losses = np.array(np_losses)
    fig = plt.figure(figsize=(8,8), facecolor='white')
    plt.ioff()
    plt.plot(np_losses)
    plt.show()
