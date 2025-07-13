from np_grad import np_grad
import torch
import numpy as np


a = np_grad(np.random.randint(0,9,(3,2)))
b = np_grad(np.random.randint(0,9,(2,5)))

a_t = torch.tensor(a, requires_grad=True)
b_t = torch.tensor(b, requires_grad=True)

out = a.exp()

out._backward(np.ones_like(out))

print(a._grad)
print(b._grad)

out = torch.exp(a_t)

out.backward(torch.ones_like(out))

print(a_t.grad)
print(b_t.grad)
