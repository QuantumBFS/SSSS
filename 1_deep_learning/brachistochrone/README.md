## Solves the brachistochrone problem with [NeuralODE](https://github.com/rtqichen/torchdiffeq)

Do you still remember the quickest descent problem as learned as a kid ? If not, refresh your memory here [here](http://mathworld.wolfram.com/BrachistochroneProblem.html). 

Here, we are going to solve the problem by minimizing the path parametrized as a neural network.The unusual thing is that here the objective function involves an integration 
$$
t = \int_{x_0}^{x_1}  \sqrt{\frac{1+(dy/dx)^2}{2 g (y_1- y_0) + v_0^2}} d x
$$
We assume the particle moves from $(x_0, y_0)=(0, 0)$ to $(x_1, y_1)$ along the path $y(x)$ . And $g$ is the gravity constant, $v_0$ is the initial velocity. 

Evaluation of the objective function amounts to integrating an ODE. And NeuralODE computes its gradient with respect to parameters of the path. 

Please play with the code and think about the following

- [ ] Change the value of $g$ and $v_0$. Does the solution agrees with your intuition ? 
- [ ] What happens when $v_0\rightarrow 0$ ?  Why does this happen ? Could you fix it ? 
- [ ] Is any other cool application you can thing of ? Welcome to share with us! 