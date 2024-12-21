import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from abc import ABC, abstractmethod

class ActivationFunction(ABC):

	@abstractmethod
	def f(x):
		pass
	
	@abstractmethod
	def df(x):
		pass


class Sigmoid(ActivationFunction):
	def f(x):
		return 1 / (1 + np.exp(-x))
	
	def df(x):
		s = self.f(x)
		return s * (1 - s)


class Network:
	def __init__(self, layer_sizes, activation_f, lr=0.1):
		self.layer_sizes = layer_sizes
		self.lr = lr
		self.activation = activation_f
		self.z = [[0 for _ in range(l)] for l in layer_sizes]
		self.b = [[0 for _ in range(l)] for l in layer_sizes]
		self.weights = []

		for l in range(len(layer_sizes)-1):
			self.weights.append(np.zeros((layer_sizes[l+1], layer_sizes[l])))


	
	def feedforward(input_data):
		for i in range(self.layer_sizes[0]):
			self.a[0][i] = self.activation.f(input_data[i])

		for l in range(len(self.weights)):
			# getting fancy
			self.a[l+1] = np.dot(self.weights[l], self.a[l])

			# apply activation function
			for i in range(len(self.a[l+1])):
				self.a[l+1][i] = self.activation.f(self.a[l+1][i])

	def train(inputs, outputs):
		return 0
		
sizes = [3, 4, 2, 5]
net = Network(sizes, Sigmoid())
print(np.shape(net.weights[1]))
