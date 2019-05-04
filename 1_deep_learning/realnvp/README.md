# Fun with Normalizing Flows

Normalizing Flows are an iterative flows from the latent space of simple *base distribution* (e.g. independent Gaussians) to the data space with complex distributions. These flows are reversible, which means that you can use them to map complex data distribution to the normal distribution, hence the name **Normalizing Flow**. Normalizing Flows are simple yet elegant generative models which demonstrate **representation learning**. 

Some background readings before start:

- Rui Shu's  [Precursor to Normalizing Flows](http://ruishu.io/2018/05/19/change-of-variables/)

- Eric Jang's tutorial  [1](https://blog.evjang.com/2018/01/nf1.html) and [2](https://blog.evjang.com/2018/01/nf2.html)

- OpenAI's [Glow](https://blog.openai.com/glow/)

Here we employ the Real NVP network introduced in [this paper](https://arxiv.org/abs/1605.08803) for variational calculation of toy target densities. The goal is to minimize the following loss 
$$
\mathcal{L} = \int d x\, q(x) [\ln q(x) + E (x)],
$$
where $q(x)$ is the model density, and $E(x)$ is a given energy function. One can show that the loss function is lower bounded $\mathcal{L} \ge -\ln Z$, where  $Z = \int d x \, e^{-E(x)}$ is the partition function. One will  arrive at the equality only when the variational density matches to the target density $q(x) = e^{-E(x)}/Z$. 

Please play with the code and finish the following tasks 

- [ ] Make a plot of the loss versus training epochs, and compare with exactly computed $-\ln Z$
- [ ] How to make sense of the learned latent space ?  Could you do something fun with it ? 

