from dumbgrad.engine import Value
import math

"""
Test derivative calculations of all the "complex" arithmetic

The way the test works is that it compares the derivative
calculated with backprop, and a numeric approximation
"""

def numeric_grad(f, x, eps=1e-6):
    return (f(x + eps) - f(x)) * (1/eps)

def fn(x, f):
    if not isinstance(x, Value):
        x = Value(x)
    y = f(x)
    return y

def chained_fn(x, f):
    if not isinstance(x, Value):
        x = Value(x)

    y = fn(x, f)
    for _ in range(100):
        y = fn(y, f)
    return y

def test_sanity():
    """
    A rather pointless test that only serves as a quick sanity check
    might also come in handy if the internals change
    """
    x = Value(0.5)

    y1 = x.tanh()
    y2 = fn(x, Value.tanh)
    assert y1.data == y2.data

    y1 = x.sigmoid()
    y2 = fn(x, Value.sigmoid)
    assert y1.data == y2.data

    y1 = x.relu()
    y2 = fn(x, Value.relu)
    assert y1.data == y2.data

    y1 = x.leaky_relu()
    y2 = fn(x, Value.leaky_relu)
    assert y1.data == y2.data

    y1 = x.log()
    y2 = fn(x, Value.log)
    assert y1.data == y2.data

def test_fn():
    def tanh(x):
        return fn(x, Value.tanh)

    def sigmoid(x):
        return fn(x, Value.sigmoid)

    def relu(x):
        return fn(x, Value.relu)

    def leaky_relu(x):
        return fn(x, Value.leaky_relu)

    def log(x):
        return fn(x, Value.log)

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
    y = fn(x, Value.relu)
    y.backprop(y.make_topo())
    numgrad = numeric_grad(relu, x).data
    assert abs(x.grad - numgrad) < 1e-3

    x.grad = 0
    y = fn(x, Value.leaky_relu)
    y.backprop(y.make_topo())
    numgrad = numeric_grad(leaky_relu, x).data
    assert abs(x.grad - numgrad) < 1e-3

    x.grad = 0
    y = fn(x, Value.log)
    y.backprop(y.make_topo())
    numgrad = numeric_grad(log, x).data
    assert abs(x.grad - numgrad) < 1e-3

def test_chained_fn():
    def tanh(x):
        return chained_fn(x, Value.tanh)

    def sigmoid(x):
        return chained_fn(x, Value.sigmoid)

    def relu(x):
        return chained_fn(x, Value.relu)

    def leaky_relu(x):
        return chained_fn(x, Value.leaky_relu)

    def log(x):
        return chained_fn(x, Value.log)

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
    y = fn(x, Value.relu)
    y.backprop(y.make_topo())
    numgrad = numeric_grad(relu, x).data
    assert abs(x.grad - numgrad) < 1e-3

    x.grad = 0
    y = fn(x, Value.leaky_relu)
    y.backprop(y.make_topo())
    numgrad = numeric_grad(leaky_relu, x).data
    assert abs(x.grad - numgrad) < 1e-3

    x = Value(3)
    y = fn(x, Value.log)
    y.backprop(y.make_topo())
    numgrad = numeric_grad(log, x).data
    assert abs(x.grad - numgrad) < 1e-3

    #TODO: test chained log somehow

if __name__ == "__main__":
    test_sanity()
    test_fn()
    test_chained_fn()
