from dumbgrad.utils import make_batches
from sklearn.datasets import load_iris
import random

def are_batches_even(x, y, batch_size):
    input_batches, output_batches = make_batches(x, y, batch_size)
    for input_batch, output_batch in zip(input_batches, output_batches):
        if len(input_batch) != batch_size and len(output_batch) != batch_size:
            return False

    return True

def factorize(n):
    factors = set()
    for i in range(1, int(n**0.5)+1):
        if n % i == 0:
            factors.add(i)
            factors.add(n // i)

    return sorted(factors)


def test_make_batches():
    dataset = load_iris()
    x, y = dataset.data.tolist(), dataset.target.tolist()
    n = len(x)

    # get all the factors of the dataset length by dividing by
    # all numbers up until the square root
    batch_sizes = factorize(n)

    for s in batch_sizes:
        assert are_batches_even(x, y, s)

def test_make_batches_uneven():
    dataset = load_iris()
    x, y = dataset.data.tolist(), dataset.target.tolist()
    n = len(x)

    # get all the factors of the dataset length by dividing by
    # all numbers up until the square root
    factors = factorize(n)
    batch_sizes = [i for i in range(1, max(factors)) if i not in factors]

    for s in batch_sizes:
        assert are_batches_even(x, y, s)

def test_make_batches_uneven_drop():
    dataset = load_iris()
    x, y = dataset.data.tolist(), dataset.target.tolist()
    n = len(x)

    # get all the factors of the dataset length by dividing by
    # all numbers up until the square root
    factors = factorize(n)
    batch_sizes = [i for i in range(1, max(factors)) if i not in factors]

    for s in batch_sizes:
        input_batches, output_batches = make_batches(x, y, s)
        total_length = 0
        for input_batch, output_batch in zip(input_batches, output_batches):
            total_length += len(input_batch)

        # assert that the batched samples and the
        # dropped samples amount the total sample count
        assert total_length + (n % s) == n



if __name__ == "__main__":
    test_make_batches()
    test_make_batches_uneven()
    test_make_batches_uneven_drop()
