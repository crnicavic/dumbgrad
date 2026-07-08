# a sanity check of backprop
from dumbgrad.engine import *
from dumbgrad.nn import *

def numeric_grad(f, x, eps=1e-6):
    return (f(x + eps) - f(x)) * (1/eps)

def loss(nn):
    x = [1, 1]
    y_pred = nn(x)
    loss = y_pred[0]**2
    return loss

def find_gradients_numeric(loss, eps=1e-6):
    topo = loss.make_topo()
    numgrads = {}
    for node in topo:
        if node.label != 'w' and node.label != 'b':
            continue
        # find the numeric derivative
        prev_loss = loss.data
        node.data += eps
        for _node in reversed(topo):
            _node.update()
        numgrads[node] = (loss.data - prev_loss) / eps

        # revert
        node.data -= eps
        for _node in reversed(topo):
            _node.update()
    return numgrads

input_count = 2
layer_sizes = [50, 10, 1]
nn = Network(input_count, layer_sizes)
loss = loss(nn)
numgrads = find_gradients_numeric(loss)
topo = loss.make_topo()
loss.backprop(topo)
for node in topo:
    if node in numgrads and abs(numgrads[node] - node.grad) > 1e-3:
        print(
        f"""
        something wrong!
        got: {node.grad}
        expected: {numgrads[node]}
        """)

g = draw_dot(loss)
g.render(filename="loss", format="png")
