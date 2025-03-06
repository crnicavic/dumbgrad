import numpy as np
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

	def __mul__(self, number):
		number = number if isinstance(number, Value) else Value(number)
		out = Value(self.data * number.data, '*', children=[self, number])
		return out

	def __rmul__(self, number):
		return self * number

	# set the gradient of children
	def backprop(self):
		if self.op == '+':
			for child in self.children:
				child.grad += self.grad # nice and simple
		elif self.op == '*':
			# shit
			product = 1
			for child in self.children:
				product *= child.data
			
			for child in self.children:
				child.grad += self.grad * product / child.data
		elif self.op == 'tanh':
			# there is only one child
			child = self.children[0]
			child.grad += (1 - self.data**2) * self.grad
		
		for child in self.children:
			child.backprop()

	def __repr__(self):
		return f"{self.label} = {self.data}, gradient = {self.grad}, op = {self.op}"
"""
a = Value(3); a.label = 'a'
b = Value(4); b.label = 'b'
c = a * b; c.label = 'c'
d = c * b; d.label = 'd'
e = d * a; e.label = 'e'
e.grad = 1
e.backprop()
print(a)
print(b)
print(c)
print(d)
print(e)
"""
