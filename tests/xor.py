from dumbgrad.engine import Value
from dumbgrad.nn import Network

n = Network(2, [5, 2])

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
n.train(x, y)
#draw_dot(loss)
y_pred = [n(i) for i in x]
n.test(x, y)
