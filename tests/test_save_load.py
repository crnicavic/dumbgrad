from dumbgrad.nn import *
from dumbgrad.utils import *
import os

def compare_networks(n1, n2):

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

    n1_parameters = n1.parameters()
    n2_parameters = n2.parameters()
    assert len(n1_parameters) == len(n2_parameters)
    for p1, p2 in zip(n1_parameters, n2_parameters):
        assert p1.data == p2.data

def test_load_save_sanity():
    n1 = Network([
        Input(5),
        Layer(5),
        LSTM(5),
        Layer(5)
    ])

    n1.build(seed=0, loss="cross_entropy")
    path = "./test_network.json"
    n1.save(path)
    n2 = Network.load(path)
    os.remove(path)

    compare_networks(n1, n2)

def test_load_save_complete():

    class_count = 5
    input_count = class_count + 1
    lstm_size = 20
    output_count = input_count

    sequences = [
        [
            [
                int(i == (j + k) % class_count)
                for i in range(input_count)
            ]
            for j in range(class_count)
        ]
        for k in range(class_count)
    ]

    inputs = sequences
    end = [int(i == input_count-1) for i in range(input_count)]
    outputs = [sequence[1:] + [end] for sequence in sequences]

    n1 = Network([
        Input(input_count),
        Layer(5, activation="tanh"),
        LSTM(lstm_size),
        Layer(5, activation="sigmoid"),
        Layer(5, activation="relu"),
        Layer(5, activation="leaky_relu"),
        Layer(output_count, activation="softmax")
    ])

    reg = L2Regularization()
    opt = Optimizer(lr=0.01)
    n1.build(seed=0, loss="cross_entropy", optimizer=opt, regularization=reg)
    path = "./test_network.json"
    n1.save(path)
    n2 = Network.load(path)
    os.remove(path)

    compare_networks(n1, n2)

    n1.train(inputs, outputs)
    n2.train(inputs, outputs)

    n1_parameters = n1.parameters()
    n2_parameters = n2.parameters()
    assert len(n1_parameters) == len(n2_parameters)
    for p1, p2 in zip(n1_parameters, n2_parameters):
        assert p1.data == p2.data

    n1.test(inputs, outputs)
    n2.test(inputs, outputs)


if __name__ == "__main__":
    test_load_save_sanity()
    test_load_save_complete()
