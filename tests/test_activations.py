from dumbgrad.engine import Value
import math

"""
Test derivative calculations of all the activation functions

The way the test works is that it compares the derivative
calculated with backprop, and a numeric approximation
"""

def numeric_grad(f, x, eps=1e-6):
    return (f(x + eps) - f(x)) * (1/eps)

def act(x, activation):
    if not isinstance(x, Value):
        x = Value(x)
    y = activation(x)
    return y

def chained_act(x, activation):
    if not isinstance(x, Value):
        x = Value(x)

    y = activation(x)
    for _ in range(100):
        y = activation(y)
    return y

def test_sanity():
    """
    A rather pointless test that only serves as a quick sanity check
    might also come in handy if the internals change
    """
    x = Value(0.5)

    y = x.tanh()
    y_act = act(x, Value.tanh)
    assert y.data == y_act.data

    y = x.sigmoid()
    y_act = act(x, Value.sigmoid)
    assert y.data == y_act.data

    y = x.relu()
    y_act = act(x, Value.relu)
    assert y.data == y_act.data

    y = x.leaky_relu()
    y_act = act(x, Value.leaky_relu)
    assert y.data == y_act.data

def test_act():
    def tanh(x):
        return act(x, Value.tanh)

    def sigmoid(x):
        return act(x, Value.sigmoid)

    def relu(x):
        return act(x, Value.relu)

    def leaky_relu(x):
        return act(x, Value.leaky_relu)

    x = Value(0.5)
    y = tanh(x)
    y.backprop(y.make_topo())
    numgrad = numeric_grad(tanh, x).data
    assert abs(x.grad - numgrad) < 1e-3

    x.grad = 0
    y = sigmoid(x)
    y.backprop(y.make_topo())
    numgrad = numeric_grad(sigmoid, x).data
    assert abs(x.grad - numgrad) < 1e-3

    x.grad = 0
    y = act(x, Value.relu)
    y.backprop(y.make_topo())
    numgrad = numeric_grad(relu, x).data
    assert abs(x.grad - numgrad) < 1e-3

    x.grad = 0
    y = act(x, Value.leaky_relu)
    y.backprop(y.make_topo())
    numgrad = numeric_grad(leaky_relu, x).data
    assert abs(x.grad - numgrad) < 1e-3

def test_chained_act():
    def tanh(x):
        return chained_act(x, Value.tanh)

    def sigmoid(x):
        return chained_act(x, Value.sigmoid)

    def relu(x):
        return chained_act(x, Value.relu)

    def leaky_relu(x):
        return chained_act(x, Value.leaky_relu)

    x = Value(0.5)
    y = tanh(x)
    y.backprop(y.make_topo())
    numgrad = numeric_grad(tanh, x).data
    assert abs(x.grad - numgrad) < 1e-3

    x.grad = 0
    y = sigmoid(x)
    y.backprop(y.make_topo())
    numgrad = numeric_grad(sigmoid, x).data
    assert abs(x.grad - numgrad) < 1e-3

    x.grad = 0
    y = act(x, Value.relu)
    y.backprop(y.make_topo())
    numgrad = numeric_grad(relu, x).data
    assert abs(x.grad - numgrad) < 1e-3

    x.grad = 0
    y = act(x, Value.leaky_relu)
    y.backprop(y.make_topo())
    numgrad = numeric_grad(leaky_relu, x).data
    assert abs(x.grad - numgrad) < 1e-3
