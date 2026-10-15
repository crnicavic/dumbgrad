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
    ">fixproblemquicklywithgalvanizedjets<",
    ">thequickbrownfoxjumpsoverthelazydog<",
    ">waxyandquiveringjocksfumblethepizza<",
    ">whenzombiesarrivequicklyfaxjudgepat<",
    ">thejaypigfoxzebraandmywolvesquacked<",
    ">wizardsjobistovexchumpsquicklyinfog<",
    ">heavyboxesperformquickwaltzesandjig<",
    ">packmytanboxwithfivedozenliquorjugs<",
    ">jackdawslovemybigsphinxmadeofquartz<",
    ">packmyboxwithfivedozenbigliquorjugs<",
]

ALPHABET = 'abcdefghijklmnopqrstuvwxyz'

def test_pangram():
    """
    Verify that all of the sentences are pangrams
    """
    uniq = unique(PANGRAMS, per_character=True)
    # the 2 are > and <
    assert len(uniq) == len(ALPHABET) + 2

def get_sequence_mapping(sequence, labels):
    """
    Helper function to reverse the mapping.
    for a sequence "abcd" and the encoding:
    [
    [1 0 0 0]
    [0 1 0 0]
    [0 0 1 0]
    [0 0 0 1]
    ]
    it figures out to what encoding each of
    the elements of the sequence is mapped to
    """
    mapping = {}
    for val, label in zip(sequence, labels):
        # if thing not in mapping, create, and checking for difference is pointless
        if val not in mapping:
            mapping[val] = label
            continue

        assert mapping[val] == label, "mapping is not consistent!"

    return mapping

def test_lstm_pangram():
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
    n.lstm_train(inputs, outputs, epochs=20)
    mapping = get_sequence_mapping(PANGRAMS[0], encoded_pangrams[0])
    reverse_mapping = {argmax(v): k for k, v in mapping.items()}
    correct_count = 0
    total_count = len(PANGRAMS[0]) * len(PANGRAMS)
    for input_string, pangram in zip(inputs, PANGRAMS):
        out = ''
        reset = True
        for char in input_string:
            out += reverse_mapping[argmax(n(char, reset))]
            reset = False

        for i in range(len(out)):
            correct_count += int(pangram[i+1] == out[i])
        print(f"expected: {pangram[1:]}")
        print(f"got: {out}")

    print(f"total accuracy: {correct_count}/{total_count}")


if __name__ == "__main__":
    test_pangram()
    test_lstm_pangram()
