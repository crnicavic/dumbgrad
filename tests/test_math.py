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

def chained_fn(x, f, count=100):
    if not isinstance(x, Value):
        x = Value(x)

    y = fn(x, f)
    for _ in range(count):
        y = fn(y, f)
    return y

def test_sanity():
    """
    A rather pointless test that only serves as a quick sanity check
    might also come in handy if the internals change
    """
    x = Value(0.5)

    y1 = x.tanh()
    y2 = math.tanh(x.data)
    assert y1.data == y2

    y1 = x.sigmoid()
    y2 = 1 / (1 + math.exp(-x.data))
    assert y1.data == y2

    y1 = x.relu()
    y2 = max(0, x.data)
    assert y1.data == y2

    y1 = x.leaky_relu()
    y2 = 0.01 * x.data if x.data < 0 else x.data
    assert y1.data == y2

    y1 = x.log()
    y2 = math.log(x.data)
    assert y1.data == y2

    y1 = x.exp()
    y2 = math.exp(x.data)
    assert y1.data == y2

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

    def exp(x):
        return fn(x, Value.exp)

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

    x.grad = 0
    y = fn(x, Value.exp)
    y.backprop(y.make_topo())
    numgrad = numeric_grad(exp, x).data
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
        return chained_fn(x, Value.log, count=2)

    def exp(x):
        return chained_fn(x, Value.exp, count=2)

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
    y = relu(x)
    y.backprop(y.make_topo())
    numgrad = numeric_grad(relu, x).data
    assert abs(x.grad - numgrad) < 1e-3

    x.grad = 0
    y = leaky_relu(x)
    y.backprop(y.make_topo())
    numgrad = numeric_grad(leaky_relu, x).data
    assert abs(x.grad - numgrad) < 1e-3

    #TODO: test chained log somehow
    x.data=2000
    x.grad = 0
    y = log(x)
    y.backprop(y.make_topo())
    numgrad = numeric_grad(log, x).data
    assert abs(x.grad - numgrad) < 1e-3

    x = Value(0.1)
    y = exp(x)
    y.backprop(y.make_topo())
    numgrad = numeric_grad(exp, x).data
    print(exp(x), y)
    assert abs(x.grad - numgrad) < 1e-3

if __name__ == "__main__":
    test_sanity()
    test_fn()
    test_chained_fn()
