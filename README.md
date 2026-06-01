# micrograd

A tiny scalar-valued autograd engine and a small neural net library built on top of it. Implementation based on Andrej Karpathy's micrograd, rebuilt as a learning exercise.

## Installation

```bash
git clone https://github.com/a1hmad23/micrograd.git
cd micrograd
pip install -e .
```

## Example usage

```python
from micrograd import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c.backward()
print(a.grad, b.grad)
```

## Training a neural net

```python
from micrograd.nn import MLP

n = MLP(3, [4, 4, 1])
# ... training loop ...
```

## Running tests

```bash
pytest test/
```

## License

MIT
