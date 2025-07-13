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
        """
        self[:] = np.array(value, dtype=np.float32)
        self._children = _children
        self._op = _op
        self._backward = lambda: None
        self._grad = np.zeros_like(self)

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
        out = np.log(self)

        out = np_grad(out, self, 'ln')

        def _backward(out_grad):
            self._grad = out_grad * 1 / self

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
        out = np_grad(out, self, 'exp')

        def _backward(out_grad):
            self._grad += np.exp(self)
        
        out._backward = _backward

        return out