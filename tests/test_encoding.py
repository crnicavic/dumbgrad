import random
import numpy as np
from dumbgrad.utils import *

def verify_encoding(test_sequence, per_character=False):
    encoded_labels = encode_labels(test_sequence, per_character=per_character)
    assert len(test_sequence) == len(encoded_labels)
    # if the per character flag is used, the result will be a 3d matrix of labels
    # which needs to be flattened
    if per_character:
        encoded_labels = [
            label for sequence in encoded_labels
            for label in sequence
        ]
        test_sequence = ''.join(test_sequence)
    # exactly 1 1 in each element?
    assert all([label.count(1) == 1 for label in encoded_labels])
    # everything else is zero?
    assert all([label.count(0) == (len(label) -1) for label in encoded_labels])
    # all the one hot labels have the same size?
    assert all([len(l1) == len(l2) for l1, l2 in zip(encoded_labels, encoded_labels[1:])])

    # check if the mapping is consistent (function does it implicitly)
    _ = get_sequence_mapping(test_sequence, encoded_labels)

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

def test_encoding_pangrams():
    pangrams = [
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
    verify_encoding(pangrams, per_character=True)

def test_encoding_strings():
    strings = [
        "Alpine Linux",
        "AlmaLinux",
        "Amazon Linux",
        "Antergos",
        "antiX",
        "ArcoLinux",
        "Arch Linux",
        "Asahi Linux",
        "BackTrack",
        "Bedrock Linux",
        "BlackArch",
        "Bodhi Linux",
        "CachyOS",
        "CentOS",
        "CentOS Stream",
        "ChromeOS",
        "Clear Linux",
        "CRUX",
        "Debian",
        "deepin",
        "Devuan",
        "Edubuntu",
        "elementary OS",
        "EndeavourOS",
        "Fedora",
        "Fedora Silverblue",
        "Fire OS",
        "Garuda Linux",
        "Gentoo Linux",
        "GNU Guix System",
        "GrapheneOS",
        "KDE neon",
        "Kali Linux",
        "Knoppix",
        "Kubuntu",
        "Kylin",
        "LibreSphere",
        "LineageOS",
        "Linux From Scratch",
        "Linux Mint",
        "Linux Mint Debian Edition",
        "Lubuntu",
        "Mageia",
        "Mandriva Linux",
        "Manjaro",
        "MX Linux",
        "Nitrux",
        "NixOS",
        "openKylin",
        "openSUSE Leap",
        "openSUSE Tumbleweed",
        "Oracle Linux",
        "Parabola GNU/Linux-libre",
        "Parrot OS",
        "Pentoo",
        "Peppermint OS",
        "Pop!_OS",
        "Porteus",
        "Puppy Linux",
        "PureOS",
        "Qubes OS",
        "Red Hat Enterprise Linux",
        "Rocky Linux",
        "ROSA Linux",
        "Salix",
        "Scientific Linux",
        "Slackware",
        "Slax",
        "SteamOS",
        "SUSE Linux Enterprise",
        "Tails",
        "Trisquel GNU/Linux",
        "Ubuntu",
        "Ubuntu MATE",
        "Ubuntu Studio",
        "Ubuntu Unity",
        "Ultramarine Linux",
        "Void Linux",
        "VyOS",
        "Xubuntu",
        "Zentyal",
        "Zorin OS",
    ]
    verify_encoding(strings, per_character=True)

if __name__ == "__main__":
    test_encoding_sanity()
    test_encoding_integers()
    test_encoding_sorting()
    test_encoding_pangrams()
    test_encoding_strings()
