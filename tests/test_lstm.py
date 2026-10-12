from dumbgrad.nn import *
from dumbgrad.utils import *

def create_lstm(lstm_size, input_count):
    lstm = LSTM(lstm_size)
    lstm.build(input_count, rng=random.Random(0))
    return lstm

def test_lstm_weights():
    lstm = create_lstm(10, 5)
    weights = lstm.weights()
    assert all([isinstance(w, Parameter) for w in weights])

def test_lstm_parameters():
    lstm = create_lstm(10, 5)
    parameters = lstm.parameters()
    assert all([isinstance(p, Parameter) for p in parameters])

def test_lstm_sanity():
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
    n.build(seed=0, optimizer=opt)
    n.lstm_train(inputs, outputs)

    for input_sequence, output_sequence in zip(inputs, outputs):
        n.layers[0].c_t[:] = [Value(0) for _ in range(lstm_size)]
        n.layers[0].h_t[:] = [Value(0) for _ in range(lstm_size)]
        for x, y in zip(input_sequence, output_sequence):
            out = n(x)
            print(out, y)
            assert all([abs(p.data - o) < 0.5 for p, o in zip(out, y)])

if __name__ == "__main__":
    test_lstm_weights()
    test_lstm_parameters()
    test_lstm_sanity()
    test_lstm_train()
