# test/test_nn.py
import pytest
from micrograd.engine import Value
from micrograd.nn import Neuron, Layer, MLP


# ============================================================
# NEURON TESTS
# ============================================================

def test_neuron_output_is_value():
    n = Neuron(3)
    out = n([1.0, 2.0, 3.0])
    assert isinstance(out, Value)


def test_neuron_output_in_tanh_range():
    # Output uses tanh, so it must be in (-1, 1)
    n = Neuron(3)
    out = n([1.0, 2.0, 3.0])
    assert -1.0 <= out.data <= 1.0


def test_neuron_parameter_count():
    # A neuron with nin inputs has nin weights + 1 bias
    n = Neuron(5)
    assert len(n.parameters()) == 6


def test_neuron_parameters_are_values():
    n = Neuron(3)
    for p in n.parameters():
        assert isinstance(p, Value)


# ============================================================
# LAYER TESTS
# ============================================================

def test_layer_single_neuron_returns_value():
    # When nout == 1, __call__ returns a Value directly (not a list)
    layer = Layer(3, 1)
    out = layer([1.0, 2.0, 3.0])
    assert isinstance(out, Value)


def test_layer_multi_neuron_returns_list():
    layer = Layer(3, 4)
    out = layer([1.0, 2.0, 3.0])
    assert isinstance(out, list)
    assert len(out) == 4
    assert all(isinstance(o, Value) for o in out)


def test_layer_parameter_count():
    # Layer(3, 4): 4 neurons, each with 3+1=4 params  =>  16 total
    layer = Layer(3, 4)
    assert len(layer.parameters()) == 16


# ============================================================
# MLP TESTS
# ============================================================

def test_mlp_forward_single_output():
    mlp = MLP(3, [4, 4, 1])
    out = mlp([1.0, 2.0, 3.0])
    assert isinstance(out, Value)


def test_mlp_forward_multi_output():
    mlp = MLP(3, [4, 4, 2])
    out = mlp([1.0, 2.0, 3.0])
    assert isinstance(out, list)
    assert len(out) == 2


def test_mlp_parameter_count():
    # MLP(3, [4, 4, 1]):
    #   Layer 1: 4 neurons * (3 weights + 1 bias) = 16
    #   Layer 2: 4 neurons * (4 weights + 1 bias) = 20
    #   Layer 3: 1 neuron  * (4 weights + 1 bias) = 5
    #   Total: 41
    mlp = MLP(3, [4, 4, 1])
    assert len(mlp.parameters()) == 41


def test_mlp_backward_propagates_to_all_params():
    """After backward(), at least some parameters should have non-zero gradients."""
    mlp = MLP(3, [4, 4, 1])
    out = mlp([1.0, 2.0, 3.0])
    out.backward()
    assert any(p.grad != 0.0 for p in mlp.parameters())


def test_mlp_can_learn():
    """End-to-end: run gradient descent steps and verify loss decreases."""
    import random
    random.seed(42)

    mlp = MLP(3, [4, 4, 1])
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    def compute_loss():
        ypred = [mlp(x) for x in xs]
        return sum((yp - yt) ** 2 for yp, yt in zip(ypred, ys))

    initial_loss = compute_loss().data

    for _ in range(20):
        loss = compute_loss()
        for p in mlp.parameters():
            p.grad = 0.0
        loss.backward()
        for p in mlp.parameters():
            p.data -= 0.05 * p.grad

    final_loss = compute_loss().data
    assert final_loss < initial_loss