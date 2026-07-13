import numpy as np
import random
from dumbgrad.engine import Value
import math
import random

def sum_of_squares(y, y_pred):
    diff = np.subtract(y, y_pred)
    loss = np.sum(np.power(diff, 2))
    return loss

def cross_entropy(y, y_pred):
    loss = 0
    for y1, y2 in zip(y, y_pred):
        if y1 == 1:
            loss += y1 * y2.log()
    pass

class Neuron:
    def __init__(self, input_count, output_count, rng=None, activation="tanh"):
        limit = np.sqrt(6 / (input_count + output_count))
        if rng is None:
            self.w = [Value(np.random.uniform(-limit, limit), label='w') for _ in range(input_count)]
        else:
            self.w = [Value(rng.uniform(-limit, limit), label='w') for _ in range(input_count)]
        self.b = Value(0,label='b')

        match activation:
            case "tanh":
                self.activation = Value.tanh
            case "sigmoid":
                self.activation = Value.sigmoid
            case "relu":
                self.activation = Value.relu
            case "leaky_relu":
                self.activation = Value.leaky_relu

    def __call__(self, x):
        # activation
        act = sum((wi * xi) for (wi, xi) in zip(self.w, x)) + self.b
        return self.activation(act)

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, size, activation="tanh"):
        self.size = size
        self.activation = activation

    def build(self, input_count, rng=None):
        self.neurons = [Neuron(input_count, self.size, rng, self.activation) for _ in range(self.size)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

# just a placeholder for prettier formating
class Input:
    def __init__(self, size):
        self.size = size

class Network:
    def __init__(self, layers):
        if not isinstance(layers[0], Input):
            raise TypeError("First layer is not an input!")
        self.layers = layers

    def build(self, seed=None, loss="sum_of_squares"):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = None

        if loss == "sum_of_squares" or loss is None:
            self.loss = sum_of_squares
        elif loss == "cross_entropy":
            self.loss = cross_entropy

        # dont build the first layer!
        for prev_layer, layer in zip(self.layers, self.layers[1:]):
            layer.build(prev_layer.size, rng)

        self.layers.pop(0)

    def __call__(self, x):
        out = [_x if isinstance(_x, Value) else Value(_x) for _x in x]
        for l in self.layers:
            out = l(out)
        return out

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def train(self, inputs, outputs, omega1=0.9, omega2=0.99, lr=0.01, epochs=100, eps=1e-6):
        """
        this builds the computation graph for loss
        over the entire dataset. Then calculate the
        gradients with backprop and use ADAM to
        update the parameters.
        Repeat for epochs amount of times.
        """
        y_pred = [self(x) for x in inputs]
        loss = self.loss(outputs, y_pred)
        topo = loss.make_topo()
        for t in range(1, epochs+1):
            #print(f"loss in epoch {t}: {loss.data}")
            loss.backprop(topo)
            for p in self.parameters():
                p.m = omega1 * p.m + (1 - omega1) * p.grad
                p.v = omega2 * p.v + (1 - omega2) * p.grad**2

                m_hat = p.m / (1 - omega1 ** t)
                v_hat = p.v / (1 - omega2 ** t)

                p.data = p.data - lr * m_hat / (math.sqrt(v_hat) + eps)
            for node in reversed(topo):
                node.update()

        return loss

    def test(self, inputs, outputs):
        y_pred = [self(x) for x in inputs]
        correct_count = 0
        for pred, output in zip(y_pred, outputs):
            if np.argmax(pred) == np.argmax(output):
                correct_count += 1

        accuracy = correct_count / len(outputs)
        print("accuracy: ", accuracy)
        return accuracy
