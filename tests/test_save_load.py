from dumbgrad.nn import *
from dumbgrad.utils import *

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

    for p1, p2 in zip(n1.parameters(), n2.parameters()):
        print(p1, p2)
        assert p1.data == p2.data


if __name__ == "__main__":
    test_load_save_sanity()
