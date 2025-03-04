class Value:
	def __init__(self, data, op=None, children=[]):
		self.data = data
		self.grad = 0 # what is the derrivative of the output by this variable
		self.children = children
		self.op = op
		self.label = ''

	def __add__(self, number):
		out = Value(self.data + number.data, '+', children=[self, number])
		return out

	def __mul__(self, number):
		out = Value(self.data * number.data, '*', children=[self, number])
		return out

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
		
		for child in self.children:
			child.backprop()

	def __repr__(self):
		return f"{self.label} = {self.data}, gradient = {self.grad}, op = {self.op}"

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
