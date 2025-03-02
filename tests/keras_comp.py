from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_digits
from abc import ABC, abstractmethod

x, y = load_digits(return_X_y=True)
x = x / 16		#normalize
num_classes = len(np.unique(y))
y = to_categorical(y, num_classes=num_classes)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False) 
sizes = [len(x[0]), 50, 50, num_classes]

model = Sequential()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.add(Dense(sizes[1], input_shape=(sizes[0],), activation='relu'))
model.add(Dense(sizes[2], activation='sigmoid'))
model.add(Dense(sizes[3], activation='relu'))
model.fit(x_train, y_train, epochs=1, batch_size=10)

_, accuracy = model.evaluate(x_test, y_test)
print(accuracy)
