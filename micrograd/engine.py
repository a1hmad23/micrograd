import math

class Value:
    def __init__(self, data, _children=(), _op='', label='', _children_labels=()):
        self.data = data
        self._prev = set(_children)
        self._children_labels = set(_children_labels)
        self._op = _op
        self.label = label
        self.grad=0.0
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) #assumes Value, Int, or Float
        out = Value(self.data + other.data, (self, other), _op='+', _children_labels=(self.label, other.label))

        def _backward():
            self.grad += out.grad # accumulates for the multivariate case and also
            other.grad += out.grad # for cases such as b = a + a
        out._backward = _backward
        return out

    def __rmul__(self, other): # fallback: if int/float * Value does not work (int__mul__(Value) !exist), then try 
        return self * other    # Value * int/float == Value.__mul__(int)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self, other), _op='*', _children_labels=(self.label, other.label))
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        return self * other**-1

    def __neg__(self):           return self * -1
    def __sub__(self, other):    return self + (-other)
    def __radd__(self, other):   return self + other
    def __rsub__(self, other):   return (-self) + other
    def __rtruediv__(self, other): return other * self**-1
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)) # int/float powers only
        out = Value(self.data**other, (self, ), f'**{other}')
        def _backward():
            self.grad += out.grad * (other * self.data**(other - 1))
        out._backward = _backward
        return out


    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), _op='exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, _children=(self, ), _op='tanh')
        def _backward():
            self.grad += out.grad * (1 - t**2)
        out._backward = _backward
        return out

    def relu(self):
        x = self.data
        t = x if x > 0 else 0
        out = Value(t, (self, ), _op='relu')
        def _backward():
            self.grad += out.grad * (x > 0)
        out._backward = _backward
        return out

    def backward(self):
        visited = set()
        topo = []
        def topsort(root):
            if root not in visited:
                visited.add(root)
                for child in root._prev:
                    topsort(child)
                topo.append(root)
                
        self.grad = 1.0
        topsort(self)
        for node in reversed(topo):
            node._backward()