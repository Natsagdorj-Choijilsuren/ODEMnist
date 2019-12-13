import torch
from torch import Tensor
import torch.nn as nn

from torch.autograd import Variable
from torch.autograd import grad

use_cuda = torch.cuda.is_available()


# Euler's method for solving 
def ode_solver(z0, t0, t1, f):

    h_max = 0.05
    n_steps = math.ceil(abs(t1 - t0)/h_max).max().item())

    h = (t1 - t0)/n_steps
    t = t0
    z = z0

    for i_step in range(n_steps):
        z = z + h*f(z,t)
        t = t + h

    return z




