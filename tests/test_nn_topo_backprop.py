# a sanity check of backprop
from dumbgrad.engine import Value
from dumbgrad.nn import Input, Layer, Network
from dumbgrad.graph import draw_dot

"""
This test compares the numeric approximation of a derivative
and the gradient calculated with backprop.

While this method is not very accurate, it is correct enough
to test the algorithm.
"""

def numeric_grad(f, x, eps=1e-6):
    return (f(x + eps) - f(x)) * (1/eps)

def make_loss(nn):
    x = [1, 1]
    y_pred = nn(x)
    loss = y_pred[0]**2
    return loss

def find_gradients_numeric(loss, nn, eps=1e-6):
    """
    This iterates through all of the parameters and
    just nudges them by eps. Then it iterates through
    the topology and runs the update method.
    Then it is subtracted from the original value and
    divided by eps.
    That result is put into the numgrad dictionary
    """
    topo = loss.make_topo()
    numgrads = {}
    for p in nn.parameters():
        # find the numeric derivative
        prev_loss = loss.data
        p.data += eps
        for node in topo:
            node.update()
        numgrads[p] = (loss.data - prev_loss) / eps

        # revert
        p.data -= eps
        for node in topo:
            node.update()
    return numgrads

def gradient_cmp(numgrads, topo):
    for node in topo:
        if node in numgrads and abs(numgrads[node] - node.grad) > 1e-3:
            print(
            f"""
            something wrong!
            got: {node.grad}
            expected: {numgrads[node]}
            """)
            return False

    return True

def test_gradients():
    nn = Network([
        Input(5),
        Layer(5),
        Layer(20),
        Layer(8)
    ])
    nn.build()
    loss = make_loss(nn)
    numgrads = find_gradients_numeric(loss, nn)
    topo = loss.make_topo()
    loss.backprop(topo)
    g = draw_dot(loss)
    g.render(filename="loss", format="png")
    assert gradient_cmp(numgrads, topo)

if __name__ == "__main__":
    test_gradients()
