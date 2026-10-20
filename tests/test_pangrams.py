"""
Testing the capabilities of LSTM's
to work with charaters by learning different phrases

The following pangrams are used:
    Fix problem quickly with galvanized jets,
    The quick brown fox jumps over the lazy dog,
    Waxy and quivering, jocks fumble the pizza,
    When zombies arrive, quickly fax judge Pat,
    The jay, pig, fox, zebra, and my wolves quacked,
    Wizard's job is to vex chumps quickly in fog,
    Heavy boxes perform quick waltzes and jig,
    Pack my tan box with five dozen liquor jugs,
    Jackdaws love my big sphinx made of quartz,
    Pack my box with five dozen big liquor jugs,

Pangrams are used so that every symbol
appears atleast once in every sentence.

All of the whitespace, commas and similiar
characters are stripped so that the phrases
remain a fixed length. Uppercase letters are
converted into lowercase to reduce the sign count
further.

Signs > and < are used to signify beggining and
end of each phrase.
"""

from dumbgrad.nn import *
from dumbgrad.utils import *

PANGRAMS = [
    "Fix problem quickly with galvanized jets",
    "The quick brown fox jumps over the lazy dog",
    "Waxy and quivering, jocks fumble the pizza",
    "When zombies arrive, quickly fax judge Pat",
    "The jay, pig, fox, zebra, and my wolves quacked",
    "Wizard's job is to vex chumps quickly in fog",
    "Heavy boxes perform quick waltzes and jig",
    "Pack my tan box with five dozen liquor jugs",
    "Jackdaws love my big sphinx made of quartz",
    "Pack my box with five dozen big liquor jugs",
]

def test_lstm_fixed_len_pangrams():
    FIXED_LEN_PANGRAMS = [
        '>' +
        p.lower()
        .replace(' ', '')
        .replace(',', '')
        .replace('\'', '')
        + '<'
        for p in PANGRAMS
    ]

    # the mappings have to be consistent between inputs and outputs
    encoded_pangrams = encode_labels(FIXED_LEN_PANGRAMS, per_character=True)
    # inputs are everything but the end character
    inputs = [encoded_pangram[:-1] for encoded_pangram in encoded_pangrams]
    # outputs is everything without the first character
    outputs = [encoded_pangram[1:] for encoded_pangram in encoded_pangrams]

    uniq = unique(FIXED_LEN_PANGRAMS, per_character=True)
    input_count = len(uniq)
    output_count = input_count
    lstm_size = 50
    n = Network([
        Input(input_count),
        LSTM(lstm_size),
        Layer(output_count)
    ])
    opt = Optimizer(lr=0.01)
    n.build(seed=0, optimizer=opt)
    n.train(inputs, outputs, epochs=50, n_workers=4)
    n.test(inputs, outputs)

def test_lstm_variable_len_pangrams():
    # the mappings have to be consistent between inputs and outputs
    encoded_pangrams = encode_labels(PANGRAMS, per_character=True)
    # inputs are everything but the end character
    inputs = [encoded_pangram[:-1] for encoded_pangram in encoded_pangrams]
    # outputs is everything without the first character
    outputs = [encoded_pangram[1:] for encoded_pangram in encoded_pangrams]

    uniq = unique(PANGRAMS, per_character=True)
    input_count = len(uniq)
    output_count = input_count
    lstm_size = 50
    n = Network([
        Input(input_count),
        LSTM(lstm_size),
        Layer(output_count)
    ])
    opt = Optimizer(lr=0.01)
    n.build(seed=0, optimizer=opt)
    n.train(inputs, outputs, epochs=50, n_workers=4)
    n.test(inputs, outputs)


if __name__ == "__main__":
    test_lstm_fixed_len_pangrams()
    test_lstm_variable_len_pangrams()
