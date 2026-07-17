from dumbgrad.engine import Value
from dumbgrad.nn import Network, Input, Layer

def test_xor():
    n = Network([
        Input(2),
        Layer(5, activation="leaky_relu"),
        Layer(2, activation="softmax")
    ])
    n.build(seed=2000, loss="cross_entropy")

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
    n.train(x, y, lr=0.08, epochs=150)
    accuracy = n.test(x, y)

    assert accuracy >= 0.99

if __name__ == "__main__":
    test_xor()
