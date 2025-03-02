import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_digits
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


# create a matrix that has the shape of proper weights matrix
def create_weights(layer_sizes):
	w = []
	for l in range(len(layer_sizes)-1):
		w.append(np.zeros((layer_sizes[l+1], layer_sizes[l])))

	return w


def non_homogenous_add(a, b):
    for i in range(len(a)):
        a[i] = np.add(a[i], b[i])
	
def non_homogenous_sub(a, b):
    for i in range(len(a)):
        a[i] = np.subtract(a[i], b[i])

def non_homogenous_divide(a, b):
    for i in range(len(a)):
        a[i] = np.divide(a[i], b)

class Network:
	def __init__(self, layer_sizes, activation_f, lr=0.1):
		self.layer_sizes = layer_sizes
		self.lr = lr
		self.activation = activation_f
		# pre activation function output of neuron
		self.z = [[0 for _ in range(l)] for l in layer_sizes]
		self.a = [[0 for _ in range(l)] for l in layer_sizes]
		self.b = [[0 for _ in range(l)] for l in layer_sizes]
		#backprop bs
		self.weights = create_weights(layer_sizes)
		
	
	def feedforward(self, input_data):
		# activation function to first layer
		for i in range(self.layer_sizes[0]):
			self.z[0][i] = input_data[i]
			#chatgpt says to not apply the transform function
			self.a[0][i] = self.z[0][i]

		for l in range(len(self.weights)):
			# getting fancy
			self.z[l+1] = np.dot(self.weights[l], self.a[l]) + self.b[l+1]

			# apply activation function
			for i in range(len(self.z[l+1])):
				self.a[l+1][i] = self.activation.f(self.z[l+1][i])
		
	def backprop(self, output):
		# ls - neuron count in each layer
		delta = [[0 for _ in range(ls)] for ls in self.layer_sizes]

		# calculate delta error for last layer
		for i in range(self.layer_sizes[-1]):
			delta[-1][i] = self.a[-1][i] - output[i]

			delta[-1][i] *= self.activation.df(self.z[-1][i])
		
		# delta error for all the other ones(skip last layer)
		for l in reversed(range(len(self.layer_sizes)-1)):
			delta[l] = np.dot(self.weights[l].transpose(), delta[l+1])
			for i in range(self.layer_sizes[l]):
				delta[l][i] *= self.activation.df(self.z[l][i])

		return delta

	def gradient(self, output):
		# storing derivatives of the cost by weights
		dw = create_weights(self.layer_sizes)

		delta = self.backprop(output)
		
		# storing derivatives of the cost by biases
		db = delta
		
		for l in range(len(self.layer_sizes) - 1):
			#NOTE: possibly reading delta from the wrong layer
			for d in range(self.layer_sizes[l+1]): # delta
				for a in range(self.layer_sizes[l]): # activation
					dw[l][d][a] = delta[l+1][d] * self.a[l][a] 

		return dw, db


	def train(self, inputs, outputs, batchsize):
		for _k in range(0, len(inputs), batchsize):
			dw = create_weights(self.layer_sizes)
			db = [[0 for _ in range(l)] for l in self.layer_sizes]
			for k in range(_k, _k + batchsize):
				if k >= len(inputs):
					return
				# TODO: calculate accuracy during training
				self.feedforward(inputs[k])
				_dw, _db = self.gradient(outputs[k])

				non_homogenous_add(dw, _dw)
				non_homogenous_add(db, _db)

			non_homogenous_divide(dw, batchsize/2)
			non_homogenous_divide(db, batchsize/2)
			non_homogenous_sub(self.weights, dw)
			non_homogenous_sub(self.b, db)


	def test(self, inputs, outputs):
		correct_count = 0
		for i in range(len(inputs)):
			self.feedforward(inputs[i])

			if np.argmax(outputs[i]) == np.argmax(self.a[-1]):
				correct_count += 1

		acc = correct_count / len(inputs) * 100
		print("Accuracy: %.2f" % acc)
		print(correct_count)

x, y = load_digits(return_X_y=True)
x = x / 16		#normalize
num_classes = len(np.unique(y))
y = to_categorical(y, num_classes=num_classes)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False) 
sizes = [len(x[0]), 30, 20, num_classes]


net = Network(sizes, Sigmoid())
net.test(x_test, y_test)
net.train(x_train, y_train, 10)
net.test(x_test, y_test)
