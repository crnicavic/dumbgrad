from dumbgrad.utils import array_split
import numpy as np

def test_split_sanity():
    array = list(range(10))
    my_split = array_split(array, 4)
    np_split = [np_list.tolist() for np_list in np.array_split(array, 4)]
    for arr in zip(my_split, np_split):
        assert my_split == np_split

def test_split():
    array = []
    for i in range(1 ,1000):
        array.append(i)
        for n in range(1, 20):
            my_split = array_split(array, n)
            np_split = [np_list.tolist() for np_list in np.array_split(array, n)]
            #print(f"my split: {my_split}")
            #print(f"np split: {np_split}")
            assert my_split == np_split

if __name__ == "__main__":
    test_split_sanity()
    test_split()
