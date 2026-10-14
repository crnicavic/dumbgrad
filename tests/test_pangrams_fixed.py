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

def test_lstm_pangram():
    # inputs are everything but the end character
    inputs = [encode_labels(p[:-1]) for p in PANGRAMS]
    # outputs is everything without the first character
    outputs = [encode_labels(p[1:]) for p in PANGRAMS]

    uniq = unique(PANGRAMS)
    input_count = len(PANGRAMS)
    output_count = input_count
    lstm_size = 10
    n = Network([
        Input(input_count),
        LSTM(lstm_size),
        Layer(output_count)
    ])
    opt = Optimizer(lr=0.01)
    n.build(seed=0, optimizer=opt)
    n.lstm_train(inputs, outputs)

if __name__ == "__main__":
    test_pangram()
    test_lstm_pangram()
