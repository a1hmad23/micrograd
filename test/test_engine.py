# test/test_engine.py
import math
import pytest
from micrograd.engine import Value


# ============================================================
# FORWARD PASS TESTS — verify .data is correct
# ============================================================

def test_addition():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    assert c.data == 5.0


def test_subtraction():
    a = Value(5.0)
    b = Value(3.0)
    c = a - b
    assert c.data == 2.0


def test_multiplication():
    a = Value(4.0)
    b = Value(3.0)
    c = a * b
    assert c.data == 12.0


def test_division():
    a = Value(6.0)
    b = Value(2.0)
    c = a / b
    assert c.data == pytest.approx(3.0)


def test_negation():
    a = Value(4.0)
    b = -a
    assert b.data == -4.0


def test_power():
    a = Value(3.0)
    b = a ** 2
    assert b.data == 9.0


def test_add_with_python_number():
    # Tests __add__ converting `2` to a Value internally
    a = Value(3.0)
    c = a + 2
    assert c.data == 5.0


def test_radd():
    # Tests __radd__: 2 + a fails on int.__add__(Value), falls back to Value.__radd__
    a = Value(3.0)
    c = 2 + a
    assert c.data == 5.0


def test_rmul():
    a = Value(3.0)
    c = 2 * a
    assert c.data == 6.0


def test_rsub():
    a = Value(3.0)
    c = 5 - a
    assert c.data == 2.0


def test_rtruediv():
    a = Value(2.0)
    c = 8 / a
    assert c.data == pytest.approx(4.0)



# ============================================================
# BACKWARD PASS TESTS — verify .grad is correct
# ============================================================

def test_backward_addition():
    # f = a + b  =>  df/da = 1, df/db = 1
    a = Value(2.0)
    b = Value(3.0)
    f = a + b
    f.backward()
    assert a.grad == 1.0
    assert b.grad == 1.0


def test_backward_multiplication():
    # f = a * b  =>  df/da = b, df/db = a
    a = Value(2.0)
    b = Value(3.0)
    f = a * b
    f.backward()
    assert a.grad == 3.0
    assert b.grad == 2.0


def test_backward_power():
    # f = a^3  =>  df/da = 3a^2 = 3 * 4 = 12
    a = Value(2.0)
    f = a ** 3
    f.backward()
    assert a.grad == 12.0


def test_backward_chained():
    # f = (a + b) * c
    # df/da = c = 4
    # df/db = c = 4
    # df/dc = a + b = 5
    a = Value(2.0)
    b = Value(3.0)
    c = Value(4.0)
    f = (a + b) * c
    f.backward()
    assert a.grad == 4.0
    assert b.grad == 4.0
    assert c.grad == 5.0


def test_gradient_accumulation_simple():
    # When `a` is used twice, gradients must accumulate.
    # f = a + a  =>  df/da = 2 (not 1!)
    a = Value(3.0)
    f = a + a
    f.backward()
    assert a.grad == 2.0


def test_gradient_accumulation_complex():
    # f = a*b + a  =>  df/da = b + 1 = 4, df/db = a = 3
    a = Value(3.0)
    b = Value(3.0)
    f = a * b + a
    f.backward()
    assert a.grad == 4.0
    assert b.grad == 3.0


def test_gradient_accumulation_multivariate():
    # a contributes to f through two separate paths:
    #   c = a * b
    #   d = a + b
    #   f = c * d = (a*b) * (a+b)
    #
    # df/da via chain rule:
    #   df/da = (df/dc)(dc/da) + (df/dd)(dd/da)
    #         = d * b       +  c * 1
    #         = (a+b)*b     +  (a*b)*1
    #         = b^2 + ab    +  ab
    #         = b^2 + 2ab
    # With a=2, b=3: df/da = 9 + 12 = 21
    #
    # df/db = (df/dc)(dc/db) + (df/dd)(dd/db)
    #       = d*a + c*1 = (a+b)*a + a*b = a^2 + ab + ab = a^2 + 2ab
    # With a=2, b=3: df/db = 4 + 12 = 16
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    d = a + b
    f = c * d
    f.backward()
    assert a.grad == 21.0
    assert b.grad == 16.0


# ============================================================
# ACTIVATION FUNCTION TESTS
# ============================================================

def test_tanh_forward():
    a = Value(0.0)
    b = a.tanh()
    assert b.data == pytest.approx(0.0)

    a = Value(1.0)
    b = a.tanh()
    assert b.data == pytest.approx(math.tanh(1.0))


def test_tanh_backward():
    # d/dx tanh(x) = 1 - tanh(x)^2
    a = Value(0.5)
    b = a.tanh()
    b.backward()
    expected = 1 - math.tanh(0.5) ** 2
    assert a.grad == pytest.approx(expected)


def test_exp_forward():
    a = Value(1.0)
    b = a.exp()
    assert b.data == pytest.approx(math.e)


def test_exp_backward():
    # d/dx exp(x) = exp(x)
    a = Value(2.0)
    b = a.exp()
    b.backward()
    assert a.grad == pytest.approx(math.exp(2.0))


def test_relu_forward_positive():
    a = Value(2.0)
    b = a.relu()
    assert b.data == 2.0


def test_relu_forward_negative():
    a = Value(-3.0)
    b = a.relu()
    assert b.data == 0.0


def test_relu_forward_zero():
    a = Value(0.0)
    b = a.relu()
    assert b.data == 0.0


def test_relu_backward_positive():
    # d/dx relu(x) = 1 when x > 0
    a = Value(2.0)
    b = a.relu()
    b.backward()
    assert a.grad == 1.0


def test_relu_backward_negative():
    # d/dx relu(x) = 0 when x <= 0
    a = Value(-2.0)
    b = a.relu()
    b.backward()
    assert a.grad == 0.0
    
# ============================================================
# COMPREHENSIVE TEST AGAINST PYTORCH
# ============================================================

def test_sanity_check_against_pytorch():
    """Build the same expression in micrograd and PyTorch, verify forward and backward agree."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed; skipping reference comparison")

    # ----- micrograd -----
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z * x
    h = (z * z).tanh()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    # ----- pytorch (reference) -----
    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z * x
    h = (z * z).tanh()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    tol = 1e-6
    assert abs(ymg.data - ypt.data.item()) < tol
    assert abs(xmg.grad - xpt.grad.item()) < tol