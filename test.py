from np_grad import np_grad
import torch
import numpy as np


a = np_grad(np.random.rand(3,2))
b = np_grad(np.random.rand(2,4))
c = np_grad([1])

a_t = torch.tensor(a, requires_grad=True)
b_t = torch.tensor(b, requires_grad=True)
c_t = torch.tensor(c, requires_grad=True)

out = c / (c + (-a).exp())

out = out.log()
out = out * a

out = out @ b

out.backward()

print(a._grad)

out = torch.sigmoid(a_t)

out = torch.log(out)
out = out * a_t

out = out @ b_t

out.backward(torch.ones_like(out))

print(a_t.grad)
