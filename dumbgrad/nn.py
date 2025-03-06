import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_digits
from abc import ABC, abstractmethod
from engine import Value
from graph import draw_dot

class ActivationFunction(ABC):

	@abstractmethod
	def f(self, x):
		pass
	
	@abstractmethod
	def df(self, x):
		pass


class Sigmoid(ActivationFunction):
	def f(self, x):
		return 1 / (1 + np.exp(-x))
	
	def df(self, x):
		s = self.f(x)
		return s * (1 - s)

class Neuron:
	def __init__(self, number_inputs):
		self.w = [Value(np.random.uniform(-1, 1)) for _ in range(number_inputs)]
		self.b = Value(0)
	
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
	
	def train(self, inputs, outputs):
		for _ in range(1):
			y_pred = [self(x) for x in inputs]
			print(y_pred, outputs)
			diff = [yout - ypred for (yout, ypred) in zip(outputs, y_pred)]
			loss = sum([d**2 for d in diff])

			loss.grad = 1
			print(loss)
			loss.backprop()
			for p in self.parameters():
				print(p)
				p.data -= 0.01 * p.grad
				print(p)


n = Network(2, [4, 1])

x = [
	[0, 0],
	[1, 0],
	[0, 1],
	[1, 1],
]

y = [0, 1, 1, 0]
n.train(x, y)
