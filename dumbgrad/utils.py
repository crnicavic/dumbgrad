import itertools

def to_categorical(y, num_classes):
    """
    Convert "regular" array to one-hot encoded matrix
    """
    batch_size = len(y)
    categorical = [[0 for _j in range(num_classes)] for _i in range(batch_size)]
    for _i, _y in enumerate(y):
        categorical[_i][_y] = 1
    return categorical

def from_categorical(y):
    """
    Convert a one-hot encoded matrix to "regular array"
    """
    regular = [0 for _ in range(len(y))]
    for i in range(len(y)):
        for j in range(len(y[i])):
            regular[i] = argmax(y[i])

    return regular
    

def normalize(x):
    x_ = flatten(x)
    min_, max_ = min(x_), max(x_)
    for i in range(len(x)):
        for j in range(len(x[i])):
            x[i][j] = (x[i][j] - min_) / (max_ - min_)

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
