# a sanity check of backprop
from dumbgrad.engine import *

def numeric_grad(f, x, eps=1e-6):
    return (f(x + eps) - f(x)) * (1/eps)

"""
These 2 cases test the case where a single node has multiple parents.
If a node gets added to the topology before both of it's parents
then the gradients will be computed wrong.

For context, both a and x are children of c, and if x gets added
to the topology before a, the gradient of x will be wrong.

The reason for that is because backprop will go through the topology,
and when it's time to find the gradient of x, the gradient of a has not
been calculated yet.
"""
def case1(x):
    if not isinstance(x, Value):
        x = Value(x)
    a = x.tanh()
    a.label = 'a'
    b = a + a
    b.label = 'b'
    c = a * x
    c.label = 'c'
    d = b + c
    d.label = 'd'
    return d

"""
This case only has a few more steps, but the same problem arises
"""
def case2(x):
    if not isinstance(x, Value):
        x = Value(x)
    a = x.tanh()
    a.label = 'a'

    c = a * x
    c.label = 'c'

    y = Value(1.0)
    w = a + y
    w.label = 'w'
    m1 = w * Value(1.0)
    m1.label = 'm1'
    m2 = m1 * Value(1.0)
    m2.label = 'm2'
    m3 = m2 * Value(1.0)
    m3.label = 'm3'

    d = m3 + c
    d.label = 'd'
    return d

def run_case(f, nx, filename=None):
    x = Value(nx)
    x.label = 'x'
    d = f(x)

    topo = d.make_topo()
    d.backprop(topo)
    numgrad = numeric_grad(f, x).data
    if filename is not None:
        g = draw_dot(d)
        g.render(filename=filename, format="png")

    if abs(x.grad - numgrad) > numgrad/1000:
        print("something wrong with backprop")
        for n in topo:
            print(n.label)
        return False
    print (f"got: {x.grad}\nexpected: {numgrad}")
    return True

def test_topo_backprop():
    print("\nCASE 1 GRADIENT RESULT:")
    assert run_case(case1, 0.6)
    print("\nCASE 2 GRADIENT RESULT:")
    assert run_case(case2, 0.6)
