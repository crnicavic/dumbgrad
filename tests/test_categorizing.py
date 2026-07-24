import random
import numpy as np
from dumbgrad.utils import *

def test_unique():
    y = [random.randint(0, 0xFFFFFFFF) for _ in range(100000)]
    y_uniq = unique(y)
    assert len(y_uniq) == len(np.unique(y))
    assert sum(list(y_uniq.values())) == len(y)
    y = [random.randint(0, 1) for _ in range (10000)]
    y_uniq = unique(y)
    assert len(y_uniq) == len(np.unique(y))
    assert sum(list(y_uniq.values())) == len(y)

def test_to_categorical():
    class_count = 20
    y = [random.randint(0, class_count-1) for _ in range(1000)]
    np_categorical = np.eye(class_count)[y]
    my_categorical = to_categorical(y, class_count)
    for i in range(len(y)):
        assert argmax(np_categorical[i]) == argmax(my_categorical[i])


def test_from_categorical():
    y = sorted([random.randint(0, 100) for _ in range(10000)])
    # y uniques
    y_uniq = unique(y)
    # y categorical
    y_c = to_categorical(y, len(y_uniq))
    # y non categorical
    y_nc = from_categorical(y_c)

    for  i in range(len(y)):
        assert y[i] == y_nc[i]

if __name__ == "__main__":
    test_unique()
    test_to_categorical()
    test_from_categorical()
