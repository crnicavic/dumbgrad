import itertools

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
    unique = {}
    for a in arr:
        if a in unique:
            unique[a] += 1
        else:
            unique[a] = 1

    return unique
