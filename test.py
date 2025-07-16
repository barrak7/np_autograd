from np_grad import np_grad
import torch
import numpy as np


a = np_grad(np.random.randint(0,9,(3,2)))
b = np_grad(np.random.randint(0,9,(3,2)))
c = np_grad(np.random.randint(0,9, 1))

a_t = torch.tensor(a, requires_grad=True)
b_t = torch.tensor(b, requires_grad=True)
c_t = torch.tensor(c, requires_grad=True)

out = b-c

out._backward(np.ones_like(out))

print(c._grad)
print(b._grad)

out = b_t-c_t

out.backward(torch.ones_like(out))

print(b_t.grad)
print(c_t.grad)