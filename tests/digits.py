from dumbgrad.engine import Value
from dumbgrad.nn import Network
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

x, y = load_digits(return_X_y=True)
x = x / 16		#normalize
num_classes = len(np.unique(y))
y = to_categorical(y, num_classes=num_classes)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False) 

n = Network(len(x[0]), [30, 30, num_classes])
n.train(x_train, y_train, epochs=150, lr=0.01)
n.test(x_test, y_test)
