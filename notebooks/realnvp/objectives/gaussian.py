from .template import Target

class Gaussian(Target):
    def __init__(self):
        super(Gaussian, self).__init__(2,'Gaussian')

    def energy(self, x):
        return (-x[:,0]**2 - 2*x[:, 1]**2 - 2.0*x[:, 0] * x[:, 1])
