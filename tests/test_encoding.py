import random
import numpy as np
from dumbgrad.utils import *

def verify_encoding(test_sequence):
    encoded_labels = encode_labels(test_sequence)
    assert len(test_sequence) == len(encoded_labels)
    # exactly 1 1 in each element?
    assert all([label.count(1) == 1 for label in encoded_labels])
    assert all([label.count(0) == (len(label) -1) for label in encoded_labels])

    # check if the mapping is consistent
    get_sequence_mapping(test_sequence, encoded_labels)

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

def test_encoding_sanity():
    test_string = "The quick brown fox jumps over the lazy dog"
    verify_encoding(test_string)

def test_encoding_integers():
    test_sequence = [4, 2, 0]
    verify_encoding(test_sequence)

def test_encoding_sorting():
    test_string = "The quick brown fox jumps over the lazy dog"
    sorted_labels = encode_labels(test_string, sort=True)
    unsorted_labels = encode_labels(test_string, sort=False)
    # if the sorted string is different than the unsorted one
    # the mappings will be different
    if test_string == sorted(test_string):
        return

    unsorted_mapping = get_sequence_mapping(test_string, unsorted_labels)
    sorted_mapping = get_sequence_mapping(test_string, sorted_labels)
    # verify that the mappings are different
    assert any([unsorted_mapping[u] != sorted_mapping[s] for u, s in zip(unsorted_mapping, sorted_mapping)])



if __name__ == "__main__":
    test_encoding_sanity()
    test_encoding_integers()
    test_encoding_sorting()
