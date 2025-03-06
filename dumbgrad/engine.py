import numpy as np
from graph import draw_dot

class Value:
	def __init__(self, data, op=None, children=[]):
		self.data = data
		self.grad = 0 # what is the derrivative of the output by this variable
		self.children = children
		self.op = op
		self.label = ''


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
		if self.op == '+':
			for child in self.children:
				child.grad += self.grad # nice and simple
		elif self.op == '*':
			self.children[0].grad += self.children[1].grad
			self.children[1].grad += self.children[0].grad
		elif self.op == 'tanh':
			self.children[0].grad += (1 - self.data**2) * self.grad
		elif self.op == '-':
			self.children[0].grad += self.grad
			self.children[1].grad -= self.grad
		elif self.op == '**':
			self.children[0].grad += self.children[1].data * self.data / self.children[0].data
			self.children[1].grad += self.data * np.log(self.children[0].data)

		for child in self.children:
			child.backprop()

	def __repr__(self):
		return f"data = {self.data}, gradient = {self.grad}, op = {self.op}"
