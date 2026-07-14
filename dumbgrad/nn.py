import numpy as np
import random
from dumbgrad.engine import Value
import math
import random
import itertools

def flatten(ndarr):
    return list(itertools.chain.from_iterable(ndarr))

def sum_of_squares(_y, _y_pred):
    y = flatten(_y)
    y_pred = flatten(_y_pred)
    diff = [(y1 - y2)**2 for y1, y2 in zip(y, y_pred)]
    loss = sum(diff)
    return loss

def cross_entropy(_y, _y_pred):
    # potential TODO:
    """
    adding a skip condition inside the for loop, such as:
    if y1 == 0:
    continue

    seems to break everything, because when constructing the entire graph,
    only certain nodes get added to the graph. This causes make_topo to not
    add nodes because not all of the parents get added into the topology
    currently has to be done like this, which is needlessly slow, but it
    actually works.
    """
    y = flatten(_y)
    y_pred = flatten(_y_pred)
    ent = []
    for y1, y2 in zip(y, y_pred):
        ent.append(-y1 * y2.log())
    loss = sum(ent)
    return loss

def l1_regularization(weights):
    total = weights[0].abs()
    for w in weights[1:]:
        total += w.abs()
    return total * 0.01

def l2_regularization(weights):
    total = weights[0] ** 2
    for w in weights[1:]:
        total += w ** 2

    return total * 0.01

class Neuron:
    def __init__(self, input_count, output_count, rng=None, activation="tanh"):
        limit = math.sqrt(6 / (input_count + output_count))
        if rng is None:
            self.w = [Value(random.uniform(-limit, limit), label='w') for _ in range(input_count)]
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
            case "softmax":
                self.activation = Value.exp

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

    def __call__(self, x):
        out = [n(x) for n in self.neurons]

        # this has to be done this way to make the smallest comp graph
        if self.activation == "softmax":
            total_act = out[0]
            for o in out[1:]:
                total_act += o
            out = [o * (total_act ** -1) for o in out]
        return out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def weights(self):
        return [w for n in self.neurons for w in n.w]

    def build(self, input_count, rng=None):
        self.neurons = [Neuron(input_count, self.size, rng, self.activation) for _ in range(self.size)]

# just a placeholder for prettier formating
class Input:
    def __init__(self, size):
        self.size = size

class Network:
    def __init__(self, layers):
        if not isinstance(layers[0], Input):
            raise TypeError("First layer is not an input!")
        self.layers = layers


    def __call__(self, x):
        out = [_x if isinstance(_x, Value) else Value(_x) for _x in x]
        for l in self.layers:
            out = l(out)
        return out

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def weights(self):
        return [w for l in self.layers for w in l.weights()]

    def build(self, seed=None, loss="sum_of_squares", regularization=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = None

        if regularization == "l1":
            self.regularization = l1_regularization
        elif regularization == "l2":
            self.regularization = l2_regularization

        if loss == "sum_of_squares" or loss is None:
            self.loss = sum_of_squares
        elif loss == "cross_entropy":
            self.loss = cross_entropy

        # dont build the first layer!
        for prev_layer, layer in zip(self.layers, self.layers[1:]):
            layer.build(prev_layer.size)

        self.layers.pop(0)

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
        if self.regularization is not None:
            loss += l2_regularization(self.weights())

        topo = loss.make_topo()
        for t in range(1, epochs+1):
            print(f"loss in epoch {t}: {loss.data}")
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
        def argmax(arr):
            m = 0
            for i in range(len(arr[1:])):
                if arr[i] > arr[m]:
                    m = i

            return i
        y_pred = [self(x) for x in inputs]
        correct_count = 0
        for pred, output in zip(y_pred, outputs):
            if argmax(pred) == argmax(output):
                correct_count += 1

        accuracy = correct_count / len(outputs)
        print(f"accuracy on {len(outputs)} test samples: {accuracy}")
        return accuracy
