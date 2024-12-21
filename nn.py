import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from abc import ABC, abstractmethod

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


class Network:
	def __init__(self, layer_sizes, activation_f, lr=0.1):
		self.layer_sizes = layer_sizes
		self.lr = lr
		self.activation = activation_f
		self.z = [[0 for _ in range(l)] for l in layer_sizes]
		self.b = [[0 for _ in range(l)] for l in layer_sizes]
		#backprop bs
		self.e = [[0 for _ in range(l)] for l in layer_sizes]
		self.weights = []

		for l in range(len(layer_sizes)-1):
			self.weights.append(np.zeros((layer_sizes[l+1], layer_sizes[l])))

	
	def feedforward(self, input_data):
		for i in range(self.layer_sizes[0]):
			self.z[0][i] = self.activation.f(input_data[i])

		for l in range(len(self.weights)):
			# getting fancy
			self.z[l+1] = np.dot(self.weights[l], self.z[l])

			# apply activation function
			for i in range(len(self.z[l+1])):
				self.z[l+1][i] = self.activation.f(self.z[l+1][i])


	def backprop(self, output):
		for i in range(layer_sizes[-1]):
			if output == i:
				self.e[-1][i] = 0.5 * (1 - self.a[-1][i]) ** 2
			else:
				self.e[-1][i] = 0.5 * (-self.a[-1][i]) ** 2
			



	def train(self, inputs, outputs):
		for i in range(len(inputs)):
			self.feedforward(inputs[i])


	def test(self, inputs, outputs):
		correct_count = 0
		for i in range(len(inputs)):
			self.feedforward(inputs[i])

			if outputs[i] == np.argmax(self.z[-1]):
				correct_count += 1

		acc = correct_count / len(inputs) * 100
		print("Accuracy: %.2f" % acc)



input_data = pd.read_csv("heart.csv")
output_data = input_data.iloc[:, -1]
input_data = input_data.iloc[:, :-1]
sizes = [len(input_data.iloc[0]), 4, 2, output_data.nunique()]
net = Network(sizes, Sigmoid())
net.test(input_data.to_numpy(), output_data.to_numpy())

