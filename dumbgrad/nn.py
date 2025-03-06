import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_digits
from abc import ABC, abstractmethod
from engine import Value

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

l = Layer(2, 3)
y = l([1, 2])
for o in y:
	o.grad = 1
	o.backprop()
	print(o)

print("\n")
for p in l.parameters():
	print(p)
