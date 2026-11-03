from dumbgrad.nn import *
from dumbgrad.utils import *

def compare_networks(n1, n2):
    for p1, p2 in zip(n1.parameters(), n2.parameters()):
        assert p1.data == p2.data

    assert n1.optimizer.omega1 == n2.optimizer.omega1
    assert n1.optimizer.omega2 == n2.optimizer.omega2
    assert n1.optimizer.lr == n2.optimizer.lr
    assert n1.optimizer.eps == n2.optimizer.eps

    assert n1.regularization.__class__.__name__ == n2.regularization.__class__.__name__
    assert n1.regularization.alpha == n2.regularization.alpha

    assert n1.loss.__name__ == n2.loss.__name__

    for l1, l2 in zip(n1.layers, n2.layers):
        assert l1.__class__.__name__ == l2.__class__.__name__
        assert l1.size == l2.size
        if not isinstance(l1, LSTM):
            assert l1.activation == l2.activation

def test_load_save_sanity():
    n1 = Network([
        Input(5),
        Layer(5),
        LSTM(5),
        Layer(5)
    ])

    n1.build(loss="cross_entropy")
    n1.save("./test_network.json")

    n2 = Network.load("./test_network.json")

    compare_networks(n1, n2)

def test_load_save_methods():
    n1 = Network([
        Input(5),
        Layer(5, activation="tanh"),
        LSTM(5),
        Layer(5, activation="sigmoid"),
        Layer(5, activation="relu"),
        Layer(5, activation="leaky_relu"),
        Layer(5, activation="softmax")
    ])

    reg = L2Regularization()
    opt = Optimizer(lr=0.01)
    n1.build(seed=0, loss="cross_entropy", optimizer=opt, regularization=reg)
    n1.save("./test_network.json")

    n2 = Network.load("./test_network.json")

    compare_networks(n1, n2)



if __name__ == "__main__":
    test_load_save_sanity()
