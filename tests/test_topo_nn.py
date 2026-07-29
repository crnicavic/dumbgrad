# Test for topology build of neural network
from dumbgrad.engine import Value
from dumbgrad.nn import Network, Input, Layer

def test_topo_nn_sanity():
    """
    A very simple test of topological sorting.
    Call make_topo and check if every node is
    added only after all of it's children are.

    In other words, call make_topo and check that
    each node in the resulting topology has all
    of it's children in the topology before it.
    The way this works is that it creates a set
    of visited nodes, and for each node, it checks
    if all of the children have been visited.

    This test checks that case of a  network
    that uses practically all of the operations
    supported by the engine, so that there is a
    sizeable amount of nodes.
    """
    input_size = 5
    nn = Network([
        Input(input_size),
        Layer(5),
        Layer(5, activation="sigmoid"),
        Layer(5, activation="relu"),
        Layer(5, activation="leaky_relu"),
        Layer(5, activation="sigmoid"),
        Layer(5, activation="softmax")
    ])

    nn.build(seed=2000)

    x = [[0 for _i in range(input_size)] for _j in range(2**input_size)]
    for i in range(2**input_size):
        for j in range(input_size):
            x[i][j] = (i >> j) & 1


    loss = sum(sum(nn(_x)) for _x in x)
    topo = loss.make_topo()
    visited = set()
    for i, n in enumerate(topo):
        for c in n.children:
            assert c in visited
        visited.add(n)

def test_topo_nn_regression():
    """
    The point of this test is to make a network
    and see if all of the parameters are
    in the topology.

    This test differs from test_sanity_check
    because parameters are explicitly checked.

    Doing this might seem redundant but it
    actually complements the first test really
    well.
    """
    input_size = 5
    nn = Network([
        Input(input_size),
        Layer(5),
        Layer(5, activation="sigmoid"),
        Layer(5, activation="relu"),
        Layer(5, activation="leaky_relu"),
        Layer(5, activation="sigmoid"),
        Layer(5, activation="softmax")
    ])

    nn.build(seed=2000)

    x = [[0 for _i in range(input_size)] for _j in range(2**input_size)]
    for i in range(2**input_size):
        for j in range(input_size):
            x[i][j] = (i >> j) & 1


    loss = sum(sum(nn(_x)) for _x in x)
    topo = loss.make_topo()

    matches = 0
    params = nn.parameters()
    for t in topo:
        for p in params:
            if id(p) == id(t):
                matches +=1

    if matches != len(params):
        print("HMMM?")
    else:
        print(f"Seems fine, the {len(params)} parameters are in the topo ({len(topo)} nodes)")

    # make pytest happy
    assert matches == len(params)

def test_topo_nn_regression_sparse():
    """
    This test does the same thing, BUT it only uses
    some of the outputs, so that not all of the nodes
    get accounted into loss

    This causes topo to not use all of the operands,
    meaning that not all of the parents of a Value
    are in the loss "graph".
    """
    input_size = 5
    nn = Network([
        Input(input_size),
        Layer(5),
        Layer(5, activation="sigmoid"),
        Layer(5, activation="relu"),
        Layer(5, activation="leaky_relu"),
        Layer(5, activation="sigmoid"),
        Layer(5, activation="softmax")
    ])

    nn.build(seed=2000)

    x = [[0 for _i in range(input_size)] for _j in range(2**input_size)]
    for i in range(2**input_size):
        for j in range(input_size):
            x[i][j] = (i >> j) & 1


    # NOTE: only one of the outputs is accounted for
    loss = sum(nn(_x)[0] for _x in x)
    topo = loss.make_topo()

    matches = 0
    params = nn.parameters()
    for t in topo:
        for p in params:
            if id(p) == id(t):
                matches +=1

    if matches != len(params):
        print(f"Only {matches} parameters are in the topo ({len(topo)} nodes)")
    else:
        print(f"Seems fine, the {len(params)} parameters are in the topo ({len(topo)} nodes)")

    # make pytest happy
    assert matches == len(params)


if __name__ == "__main__":
    test_topo_nn_sanity()
    test_topo_nn_regression()
    test_topo_nn_regression_sparse()
