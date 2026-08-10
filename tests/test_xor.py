from dumbgrad.engine import Value
from dumbgrad.nn import *

def test_xor():
    n = Network([
        Input(2),
        Layer(5, activation="leaky_relu"),
        Layer(2, activation="softmax")
    ])
    opt = Optimizer(lr=0.01)
    n.build(seed=2000, optimizer=opt)

    x = [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
    ]

    y = [
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 0]
    ]
    n.train(x, y, epochs=150)
    accuracy = n.test(x, y)

    assert accuracy >= 0.99

if __name__ == "__main__":
    test_xor()
