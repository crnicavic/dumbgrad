import random
import numpy as np
from dumbgrad.utils import *

def test_categorical_conversions():
    y = sorted([random.randint(0, 100) for _ in range(10000)])
    # y uniques
    y_uniq = unique(y)
    assert len(np.unique(y)) == len(y_uniq)
    # y categorical
    y_c = to_categorical(y, len(y_uniq))

    # y non categorical
    y_nc = from_categorical(y_c)

    for  i in range(len(y)):
        assert y[i] == y_nc[i]
