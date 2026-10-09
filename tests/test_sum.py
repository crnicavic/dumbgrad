from dumbgrad.engine import Value
from dumbgrad.nn import *
import random

"""
This test is made to verify the value_sum function.

This function has to function properly, because if it doesn't
the probabilities of softmax might not sum correctly.
"""

def cmp_sum_functions(arr, tol=1e-6):
    """
    Integral test function that cases are thrown at.
    All it does it runs both regular sum() and value_sum()
    and compares the results.

    It also creates value object of the original array just
    to avoid any errors of that kind.

    It also handles the special case of float sums, where
    a tolerance for the difference is used, because
    python's sum() has internal compensation, which
    causes assertion errors.
    """
    val_arr = [Value(a) for a in arr]
    return abs(sum(arr) - value_sum(val_arr).data) < tol

def test_sanity():
    assert cmp_sum_functions([0, 0, 0])
    assert cmp_sum_functions([0, 1, 0])
    assert cmp_sum_functions([5, 6, 1])

def test_big_sum():
    arr_size = 1000000
    assert cmp_sum_functions([random.random() for _ in range(arr_size)])


if __name__ == "__main__":
    test_sanity()
    test_big_sum()
