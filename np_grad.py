from numpy import ndarray
import numpy as np

class np_grad(ndarray):
    """
        np_grad: the numpy array wrapper that provides reverse mode auto diff.
    """
    def __new__(cls, value, *args, **kwargs):
        """
            Creates a new instance of the class.

            Since np.ndarray doesn't have an __init__ function,
            I had to override __new__ as a workaround to initializing
            the parent instance with the same value.
        """
        arr = np.array(value)
        return super(np_grad, cls).__new__(cls, arr.shape, dtype=np.float32)
    
    def __init__(self, value: ndarray, _children=(), _op=''):
        """
            Parameters
            ----------
            value: ndarray
                The matrix.
            _children: tuple[Any, Any] | tuple[]
                default: ()
                The operands that resulted in value.
            _op: str
                default: '' 
                The operation that resulted in value.
            _grad: ndarray.float
                default: 0
                The gradient of the current node.
            _backward: Callable[[out_grad], None]
                default: None
                The _backward callable that performs the chain rule of the output gradient against _children.
            _eps: float
                default: 1-e7
                Epsilon, a very small value to add to operations that don't work on 0.
        """
        self[:] = np.array(value, dtype=np.float32)
        self._children = _children
        self._op = _op
        self._backward = lambda g: None
        self._grad = np.zeros_like(self)
        self._eps = 1e-7

    def __matmul__(self, other):
        """
            Overrides the @ operator. Performs matrix multiplication between self and other.

            Calls numpy matmul on the ndarrays. Uses the output value to create a new np_grad object.

            Defines the _backward function and sets it in the newly created object.

            Returns:
                np_grad, the result of the operation.
        """
        out = super(np_grad, self).__matmul__(other)
        out = np_grad(out, (self, other), '@')

        def _backward(out_grad):
            self._grad += out_grad @ other.T
            other._grad += self.T @ out_grad
        
        out._backward = _backward
        return out

    def log(self):
        """
            Applies the natural log to each element of the Matrix.
            It calls numpy's log function.

            Returns:
                out: ndarray
                    A new instance of np_grad with the natural log of self.
        """
        out = np.log(self + self._eps)

        out = np_grad(out, (self,), 'ln')

        def _backward(out_grad):
            self._grad += out_grad * 1 / self

        out._backward = _backward
        return out

    def exp(self):
        """
            Applies the natural exponential function to self.
            It calls numpy's exp function.

            Returns:
                out: ndarray
                    The output matrix with the exp of self.
        """
        out = np.exp(self)
        out = np_grad(out, (self,), 'exp')

        def _backward(out_grad):
            self._grad += np.exp(self)
        
        out._backward = _backward

        return out

    def __pow__(self, other):
        """
            Raises self to the power of other.
            It calls ndarray.__pow__(other).

            Returns:
                out: ndarray
                    Self raised to the power of other.
        """
        out = super(np_grad, self).__pow__(other)
        out = np_grad(out, (self, other), '**')

        def _backward(out_grad):
            self._grad += out_grad * other * (self ** (other - 1))
            other._grad += np.sum(out_grad * (self ** other) * self.log())

        out._backward = _backward

        return out

    def __mul__(self, other):
        """
            Multiplies self by other.
            It calls ndarray.__mul__.

            Returns:
                out: ndarray
                    Result of the multiplcation.
        """
        out = super(np_grad, self).__mul__(other)
        out = np_grad(out, (self, other), '*')

        def _backward(out_grad):
            self._grad += out_grad * other if self.shape != (1,) else np.sum(out_grad * other)
            other._grad += out_grad * self if other.shape != (1,) else np.sum(out_grad * self)

        out._backward = _backward

        return out

    def __add__(self, other):
        """
            Implements the addition operator.
            Calls ndarray.__add__.

            Returns:
                out: ndarray
                    The result of the addition.
        """
        out = super(np_grad, self).__add__(other)
        out = np_grad(out, (self, other), '+')

        def _backward(out_grad):
            self._grad += out_grad if self.shape != (1,) else np.sum(out_grad)
            other._grad += out_grad if other.shape != (1,) else np.sum(out_grad)

        out._backward = _backward

        return out

    def __neg__(self):
        """
            Negates self.
            Calls ndarray.__neg__()

            Returns:
                out: ndarray
                    Negated self.
        """
        out = super(np_grad, self).__neg__()
        out = np_grad(out, (self,), '-')

        def _backward(out_grad):
            self._grad -= out_grad

        out._backward = _backward

        return out

    def __sub__(self, other):
        """
            Implements subtraction.
            Calls ndarray.__sub__.

            Returns:
                out: ndarray
                    Result of __add__ call.
        """
        out = super(np_grad, self).__sub__(other)
        out = np_grad(out, (self, other), '-')

        def _backward(out_grad):
            self._grad += out_grad if self.shape != (1,) else np.sum(out_grad)
            other._grad += -out_grad if other.shape != (1,) else np.sum(-out_grad)

        out._backward = _backward

        return out

    def __truediv__(self, other):
        """
            Implements division.
            Calls ndarray.__div__.
            Adds epsilon to other to avoid division by zero.

            Returns:
                out: ndarray
                    The result of the division.
        """
        out = super(np_grad, self).__truediv__(other + other._eps)
        out = np_grad(out, (self, other), '/')

        def _backward(out_grad):
            self._grad += out_grad / other if self.shape != (1,) else np.sum(out_grad / other)
            other._grad += out_grad * (-self) / (other ** 2) if other.shape != (1,) else np.sum(out_grad * (-self) / (other ** 2))

        out._backward = _backward
        return out
