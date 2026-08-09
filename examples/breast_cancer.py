from dumbgrad.engine import Value
from dumbgrad.nn import Network, Input, Layer, flatten
import numpy as np
from dumbgrad.utils import *
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()
x, y = dataset.data.tolist(), dataset.target.tolist()
x = normalize(x, per_column=True)
num_classes = len(unique(y))
y = to_categorical(y, num_classes=num_classes)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=0)

n = Network([
    Input(len(x[0])),
    Layer(30),
    Layer(30),
    Layer(num_classes, activation="softmax")
])
n.build(seed=0, loss="cross_entropy", regularization="l2")
n.train(x_train, y_train, epochs=15)
n.test(x_test, y_test)
