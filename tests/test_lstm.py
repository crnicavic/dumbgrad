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

if __name__ == "__main__":
    test_lstm_sanity()
