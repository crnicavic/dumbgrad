import itertools
from math import ceil, floor

def to_categorical(y, num_classes):
    """
    Convert "regular" array to one-hot encoded matrix

    Something like:
    [4, 2, 0]

    Will be converted to:
    [
      [0 0 0 0 1],
      [0 0 1 0 0],
      [1 0 0 0 0]
    ]
    """
    batch_size = len(y)
    categorical = [[0 for _j in range(num_classes)] for _i in range(batch_size)]
    for _i, _y in enumerate(y):
        categorical[_i][_y] = 1
    return categorical

def from_categorical(y):
    """
    Convert a one-hot encoded matrix to "regular array"

    Something like:
    [
      [0 0 0 0 1],
      [0 0 1 0 0],
      [1 0 0 0 0]
    ]
    Will be converted to:
    [4, 2, 0]
    """
    return [argmax(row) for row in y]
    

def normalize(x, per_column=False):
    """
    Iterate through the entirety of the input and normalize
    in accordance to the formula:
    (x - lo) / (hi - lo)
    where:
     - lo is the minimum of the target
     - hi is the maximum of the target

    If the per column flag is True, the hi and lo
    are calculated per column, and then each column
    is normalized separetely.
    Useful when each feature has it's own magnitude of size.
    """
    scale = lambda val, lo, hi: (val - lo) / (hi - lo)
    if per_column == True:
        # list(zip(*x)) inverts the rows and columns
        # in other words it gives the transpose
        transposed = list(zip(*x))
        los = [min(col) for col in transposed]
        his = [max(col) for col in transposed]
        return [[scale(x_, lo, hi) for x_, lo, hi in zip(row, los, his)] for row in x]
    else:
        lo = min(flatten(x))
        hi = max(flatten(x))
        return [[scale(x_, lo, hi) for x_ in row] for row in x]

def flatten(ndarr):
    return list(itertools.chain.from_iterable(ndarr))

def argmax(arr):
    m = 0
    for i in range(1, len(arr)):
        if arr[i] > arr[m]:
            m = i
    return m

def unique(arr):
    """
    Returns a dictionary with each unique value
    in an array, where each key is the value of the
    array, and the dictionary value is the number of
    occurances

    For an array:
    [1 1 4 5 9 1 9]
    return dictionary would be:
    {
    1: 3
    4: 1
    5: 1
    9: 2
    }
    """
    unique = {}
    for a in arr:
        if a in unique:
            unique[a] += 1
        else:
            unique[a] = 1

    return unique

def make_batches(inputs, outputs, batch_size):
    """
    Split the dataset into batches.
    if the sample count is not divisible by
    the batch size, the last batch will be
    a different size then the rest, and
    as such it will be dropped.

    for an array:
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    and batch_size=2 this returns:
    [[10, 9], [8, 7], [6, 5], [4, 3], [2, 1]]

    for an array:
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    and batch_size=3 this returns:
    [[10, 9, 8], [7, 6, 5], [4, 3, 2]]

    Note that the 1 was dropped
    """
    batches = []
    for start in range(0, len(outputs), batch_size):
        stop = start + batch_size
        # drop uneven batch
        if stop > len(outputs):
            break
        batch = (list(inputs[start:stop]), list(outputs[start:stop]))
        batches.append(batch)
    return batches

def array_split(array, n):
    """
    Create equal chunks of an array.
    for an array:
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    split into 2 parts will return:
    [[10, 9, 8, 7, 6], [5, 4, 3, 2, 1]]

    In case of an uneven split, the algorithm
    will distribute the the remainder in a best effort manner

    for an array:
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    split into 4 parts, the split will be [3, 3, 2, 2]:
    [[10, 9, 8], [7, 6, 5], [4, 3], [2, 1]]

    This differs from batching in a way that
    this creates n arrays that are the same size,
    and batching creates arrays that have a length
    of n.
    """
    chunk, rem = divmod(len(array), n)
    # "distribute" the remainder of elements
    section_sizes = [chunk+1 if i < rem else chunk for i in range(n)]
    split_array = []
    total = 0
    for section_size in section_sizes:
        if section_size != 0:
            split_array.append(array[total:total+section_size])
        else:
            split_array.append([])
        total += section_size

    return split_array
