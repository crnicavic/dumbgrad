from dumbgrad.nn import *
from dumbgrad.utils import *

def test_lstm_sanity():
    input_count = 5
    lstm_size = 10
    lstm = LSTM(lstm_size)
    lstm.build(input_count, rng=random.Random(0))
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

    x = sequences
    y = [
        sequence[1:] + [[0 for _ in range(input_count)]]
    for sequence in sequences]

    n = Network([
        Input(input_count),
        LSTM(lstm_size),
        Layer(output_count)
    ])

    n.build(seed=0)
    n.lstm_train(x, y)

if __name__ == "__main__":
    test_lstm_train()
    test_lstm_sanity()
