# np_autograd
A numpy backward auto differentiation wrapper.


# Definitions: 

### Derivative:

A derivative is the amount by which a variable influences the output of a function at a given point as it changes slightly.

It can be expressed mathematically as:
$$
\frac{df}{dx} = \lim_{x\to 0} \frac{f(x + h) - f(x)}{h}
$$

### Automatic Differentiation

Automatic differentiation is the process of tracking all variables involved in a series of calculations in a `computational graph`. We can then find the derivative of a variable w.r.t all prior variables and functions leading up to it by applying the chain rule back through the said graph.

This can be done by defining the derivative of the basic algebraic operations that can be performed on our variable. The basic operations can then be used to compose more complex functions.

Unlike numerical differentiation which can suffer from roundoff errors, and symbolic differentiation which is hard to implement and is expensive, auto diff is much easier to implement and is less error prone.

# Motivation

Matrices are a building block of Deep Learning. Understanding how to differentiate and backpropagate through them is a fundemental requirement.

Such project will prove useful in building different types of Deep Learning models from scratch.

# Implementation

np_grad will inherit from and extend the functionality of numpy.ndarray. Thus, all operations possible on numpy.ndarray will also be possible on np_grad.

> [!NOTE]
> Not all operations will be differentiable.

Every operation will create a new instance of np_grad which stores the operands and operations that resulted in it.

It will support operations with all the datatypes which numpy.ndarray supports, although backpropagating through different data types wouldn't be possible.

# Usage:

## Example:
```py
from np_grad import np_grad

a = np_grad(np.random.rand(3, 2))
b = np_grad(np.random.rand(2, 4))
c = np_grad(np.random.rand(1))
d = np_grad(np.random.rand(1))

out = a @ b

out = out / c

out = out ** d

out.backward()

print(a._grad)
```

## Extension example:

```py
import numpy as np
from np_grad import np_grad

def cos(x):
    re = np.cos(x)                   # perform calculation
    re = np_grad(re, (x,))           # initiat np_grad obj with input as child !Important

    def _backward(out_grad):         # define backward step
        x._grad += -np.sin(x) * out_grad

    re._backward = _backward         # set result obj backward step
    return re                        # return result
```