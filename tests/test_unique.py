import random
import numpy as np
import string
from dumbgrad.utils import *

"""
Test that the unique function does that
it is supposed to. The main method of
testing it is by comparing with the
unique method from numpy
"""
def compare_unique_functions(y):
    y_uniq = unique(y)
    # compare amount of recognized values
    assert len(y_uniq) == len(np.unique(y))
    # compare that nothing was left out
    assert sum(list(y_uniq.values())) == len(y)

def test_unique_ints():
    # generate a lot of ints
    y = [random.randint(0, 0xFFFFFFFF) for _ in range(100000)]
    compare_unique_functions(y)
    y = [random.randint(0, 1) for _ in range (10000)]
    compare_unique_functions(y)

def test_unique_floats():
    y = [random.random() for _ in range(100000)]
    compare_unique_functions(y)
    y = [random.random() * 10000 for _ in range (10000)]
    compare_unique_functions(y)

def test_unique_string():
    # generate array of random characters
    str_lens = 1000
    random_chars = random.choices(string.ascii_lowercase, k=str_lens)
    y = ''.join(random_chars)
    compare_unique_functions(random_chars)

    # this has to be asserted manually since
    # numpy unique doesnt support strings
    y_uniq = unique(y)
    assert sum(list(y_uniq.values())) == len(y)

def test_unique_list_strings():
    str_lens = 10000
    str_count = 100
    # generate a list of random strings
    random_strings = [
        ''.join(random.choices(string.ascii_lowercase, k=str_lens))
        for _ in range(str_count)
    ]
    compare_unique_functions(random_strings)

    # test that the per_character flag works
    random_strings_uniq = unique(random_strings, per_character=True)
    total_len = sum([len(random_string) for random_string in random_strings])
    assert sum(list(random_strings_uniq.values())) == total_len

if __name__ == "__main__":
    test_unique_ints()
    test_unique_floats()
    test_unique_string()
    test_unique_list_strings()
