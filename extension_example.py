import numpy as np
from np_grad import np_grad


def cos(x):
    re = np.cos(x)
    re = np_grad(re, (x,))

    def _backward(out_grad):
        x._grad += -np.sin(x) * out_grad

    re._backward = _backward
    return re
