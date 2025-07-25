from np_grad import np_grad
import torch
import numpy as np
from extension_example import cos


a = np_grad(np.random.rand(3,2))
b = np_grad(np.random.rand(2,4))
c = np_grad([1])

a_t = torch.tensor(a, requires_grad=True)
b_t = torch.tensor(b, requires_grad=True)
c_t = torch.tensor(c, requires_grad=True)

out = cos(a)

out.backward()

print(a._grad)

out = torch.cos(a_t)

out.backward(torch.ones_like(out))

print(a_t.grad)
