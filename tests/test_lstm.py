"""
Tests to verify basic LSTM functions
"""
from dumbgrad.nn import *
from dumbgrad.utils import *

def create_lstm(lstm_size, input_count):
    """
    A helper for the smaller tests
    that just calls the 2 functions
    """
    lstm = LSTM(lstm_size)
    lstm.build(input_count, rng=random.Random(0))
    return lstm

def test_lstm_weights():
    """
    Small test that creates an LSTM and checks
    if all the returned weights are in one array
    """
    lstm = create_lstm(10, 5)
    weights = lstm.weights()
    assert all([isinstance(w, Parameter) for w in weights])

def test_lstm_parameters():
    """
    Basically does the same thing as the
    weights test, but it also adds the biases
    """
    lstm = create_lstm(10, 5)
    parameters = lstm.parameters()
    assert all([isinstance(p, Parameter) for p in parameters])

def test_lstm_sanity():
    """
    Another in the suite of basic tests.
    All it does it creates an LSTM and
    checks if it outputs numbers.

    The sequence is alternating
    [1, 1, 1, 1, 1] and [0, 0, 0, 0, 0]
    it doesnt do any training it just spits out numbers.

    The dimensionality of the output of the LSTM is also
    checked. The reason for this is to check the most
    basic of functionalities.
    """
    input_count = 5
    lstm_size = 10
    lstm = create_lstm(lstm_size, input_count)
    sequences = [
        [
            [Value(0) for _ in range(input_count)],
            [Value(1) for _ in range(input_count)],
            [Value(0) for _ in range(input_count)],
            [Value(1) for _ in range(input_count)],
        ]
    ]
    for sequence in sequences:
        for x in sequence:
            lstm(x)
            assert lstm_size == len(lstm.h_t)

def test_lstm_train():
    """
    Test the LSTM's ability to train on 2 sequences:
    [
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0]
            [0, 0, 0, 0, 0, 1]
        ],
        [
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0]
            [0, 0, 0, 0, 0, 1]
        ]
    ]
                            ^
                            end sequence? column

    Which are just sequences of shift left and shift right,
    but notice that the last column is ignored, and that is
    because it is the end class. It is set to 1 only on the
    final sample of the sequence.
    """
    class_count = 5
    input_count = class_count + 1
    lstm_size = 10
    output_count = input_count

    sequences = [
        [[int(i == j) for i in range(input_count)] for j in range(class_count)],
        [[int(i == (class_count-1-j)) for i in range(input_count)] for j in range(class_count)]
    ]

    inputs = sequences
    end = [int(i == input_count-1) for i in range(input_count)]
    outputs = [sequence[1:] + [end] for sequence in sequences]

    n = Network([
        Input(input_count),
        LSTM(lstm_size),
        Layer(output_count, activation="softmax")
    ])

    opt = Optimizer(lr=0.1)
    # ensure that the created network is NOT GOOD!
    is_network_good = True
    seed = -1
    while is_network_good:
        seed += 1
        n.build(seed=seed, loss="cross_entropy", optimizer=opt)
        # check if the network is actually bad
        is_network_good = n.test(inputs, outputs) > 0.5
    # train
    n.train(inputs, outputs, epochs=15, n_workers=2)
    assert n.test(inputs, outputs) >= 1.0
    # since the network was bad, training worked!
    print("Network learned something!")

def test_lstm_countup():
    """
    A bit more of an extensive training set to further test
    basic lstm capabilites. The dataset is just "counting up":

    [
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ],
        [
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        [
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        [
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        [
            [0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    ]
                            ^
                            end sequence? column

    The counting up rolls over once it's at the end of
    the array. After all of the "counting" is done,
    the end sample is used, which sets the ignored
    position to 1.
    """
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

    n = Network([
        Input(input_count),
        LSTM(lstm_size),
        Layer(output_count, activation="softmax")
    ])

    opt = Optimizer(lr=0.1)
    # ensure that the created network is NOT GOOD!
    is_network_good = True
    seed = -1
    while is_network_good:
        seed += 1
        n.build(seed=seed, loss="sum_of_squares", optimizer=opt)
        # check if the network is actually bad
        is_network_good = n.test(inputs, outputs) > 0.5
    # train
    n.train(inputs, outputs, epochs=15, n_workers=4)
    assert n.test(inputs, outputs) >= 1.0
    # since the network was bad, training worked!
    print("Network learned something!")

if __name__ == "__main__":
    test_lstm_weights()
    test_lstm_parameters()
    test_lstm_sanity()
    test_lstm_train()
    test_lstm_countup()
