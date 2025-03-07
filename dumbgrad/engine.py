import numpy as np
from graph import draw_dot

class Value:
	def __init__(self, data, op=None, children=[], label=''):
		self.data = data
		self.grad = 0 # what is the derrivative of the output by this variable
		self.children = children
		self.op = op
		self.label = label

	def tanh(self):
		out = Value(np.tanh(self.data), 'tanh', children=[self])
		return out

	def __add__(self, number):
		number = number if isinstance(number, Value) else Value(number)
		out = Value(self.data + number.data, '+', children=[self, number])
		return out

	def __radd__(self, number):
		return self + number

	def __sub__(self, number):
		number = number if isinstance(number, Value) else Value(number)
		out = Value(self.data - number.data, '-', children=[self, number])
		return out

	def __rsub__(self, number):
		number = number if isinstance(number, Value) else Value(number)
		return number - self

	def __mul__(self, number):
		number = number if isinstance(number, Value) else Value(number)
		out = Value(self.data * number.data, '*', children=[self, number])
		return out

	def __rmul__(self, number):
		return self * number

	def __pow__(self, number):
		number = number if isinstance(number, Value) else Value(number)
		out = Value(self.data ** number.data, '**', children=[self, number])
		return out

	# set the gradient of children
	def backprop(self):
		self.grad = 1
		topo = []
		visited = set()
		stack = [self]

		# **Iterative Topological Sorting**
		while stack:
			node = stack.pop()
			if node not in visited:
				visited.add(node)
				topo.append(node)
				stack.extend(node.children)  # Push children for processing

		for node in topo:
			match node.op:
				case '+':
					node.children[0].grad += node.grad
					node.children[1].grad += node.grad
				case '*':
					node.children[0].grad += node.children[1].data * node.grad
					node.children[1].grad += node.children[0].data * node.grad
				case 'tanh':
					node.children[0].grad += (1 - node.data**2) * node.grad
				case '-':
					node.children[0].grad += node.grad
					node.children[1].grad -= node.grad
				case '**':
					#TODO: make this expression shorter
					node.children[0].grad += node.children[1].data * (node.children[0].data ** (node.children[1].data -1)) * node.grad

	def __repr__(self):
		return f"data = {self.data}, gradient = {self.grad}, op = {self.op}"
