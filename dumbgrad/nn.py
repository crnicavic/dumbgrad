import numpy as np
from engine import Value

class Neuron:
	def __init__(self, number_inputs):
		self.w = [Value(np.random.uniform(-1, 1), label='w') for _ in range(number_inputs)]
		self.b = Value(0,label='b')
	
	def __call__(self, x):
		act = sum((wi * xi) for (wi, xi) in zip(self.w, x)) + self.b
		return act.tanh()

	def parameters(self):
		return self.w + [self.b]


class Layer:
	def __init__(self, number_inputs, number_outputs):
		self.neurons = [Neuron(number_inputs) for _ in range(number_outputs)]
	
	def __call__(self, x):
		out = [n(x) for n in self.neurons]
		return out

	def parameters(self):
		return [p for n in self.neurons for p in n.parameters()]

class Network:
	def __init__(self, number_inputs, layer_sizes):
		sz = [number_inputs] + layer_sizes
		self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(sz)-1)]
		

	def __call__(self, x):
		for l in self.layers:
			x = l(x)
		return x

	def parameters(self):
		return [p for l in self.layers for p in l.parameters()]
	
	def zero_grad(self):
		for p in self.parameters():
			p.grad = 0

	def train(self, inputs, outputs, omega1=0.9, omega2=0.99, lr=0.05, epochs=100, eps = 1e-3):
		"""
		 this builds the entire graph of the network
		 and since i am calculating the loss for all
		 of the inputs at once all that needs to be done
		 is to just call update() for all of the nodes
		 and then backprop
		"""
		y_pred = [self(x) for x in inputs]
		diff = np.subtract(outputs, y_pred)
		loss = np.sum(np.power(diff, 2))
		topo = loss.make_topo()
		for _ in range(epochs):

			print(loss)
			loss.backprop(topo)
			for p in self.parameters():
				p.m = omega1 * p.m + (1 - omega1) * p.grad
				p.v = omega2 * p.v + (1 - omega2) * p.grad**2

				m_hat = p.m / (1 - omega1)
				v_hat = p.v / (1 - omega2)

				p.data = p.data - lr * m_hat / (np.sqrt(v_hat) + eps)
			for node in reversed(topo):
				node.update()
			
		return loss

