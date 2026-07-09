from dumbgrad.engine import Value
from dumbgrad.nn import Network

n = Network(2, [5, 2], seed=100)

x = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
]

y = [
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
]
n.train(x, y, lr=0.08, epochs=150)
y_pred = [n(i) for i in x]
n.test(x, y)
