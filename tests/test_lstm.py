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
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ],
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0]
        ]
    ]

    Which are just sequences of shift left and shift right.
    To signalize the end of a sequence [0, 0, 0, 0, 0] is used.
    These sequences are used as the inputs for the lstm, whilst
    for the outputs, the first array is thrown out, and
    [0, 0, 0, 0, 0] is added at the end.

    Since this is used to test the sanity of the training,
    the test_by_hand() helper function is called twice.
    First, to determine if an untrained network is really bad,
    and after training to check if the network actually learned
    something.
    """
    def test_by_hand():
        for input_sequence, output_sequence in zip(inputs, outputs):
            n.layers[0].c_t[:] = [Value(0) for _ in range(lstm_size)]
            n.layers[0].h_t[:] = [Value(0) for _ in range(lstm_size)]
            for x, y in zip(input_sequence, output_sequence):
                out = n(x)
                if not all([abs(p.data - o) < 0.5 for p, o in zip(out, y)]):
                    return False
        return True
    input_count = 5
    lstm_size = 10
    output_count = input_count

    sequences = [
        [[int(i == j) for i in range(input_count)] for j in range(input_count)],
        [[int(i == (input_count-1-j)) for i in range(input_count)] for j in range(input_count)]
    ]

    inputs = sequences
    outputs = [
        sequence[1:] + [[0 for _ in range(input_count)]]
    for sequence in sequences]

    n = Network([
        Input(input_count),
        LSTM(lstm_size),
        Layer(output_count)
    ])

    opt = Optimizer(lr=0.01)
    # ensure that the created network is NOT GOOD!
    is_network_good = True
    seed = 0
    while is_network_good:
        n.build(seed=0, optimizer=opt)
        # check if the network is actually bad
        is_network_good = test_by_hand()
        seed += 1
    # train
    n.lstm_train(inputs, outputs)
    # if the assertion passes the network is fine
    assert test_by_hand()
    # since the network was bad, then training worked!
    print("Network learned something!")


if __name__ == "__main__":
    test_lstm_weights()
    test_lstm_parameters()
    test_lstm_sanity()
    test_lstm_train()
