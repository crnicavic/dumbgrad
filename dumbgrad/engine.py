import numpy as np
import math
from dumbgrad.graph import draw_dot

class Value:
    def __init__(self, data, op=None, children=[], label=''):
        self.data = data
        self.grad = 0 # what is the derrivative of the output by this variable
        self.parents = []
        self.children = children
        self.op = op
        self.label = label
        self.m = 0
        self.v = 0
        for c in children:
            c.parents.append(self)

    def __add__(self, number):
        number = number if isinstance(number, Value) else Value(number)
        out = Value(self.data + number.data, '+', children=[self, number])
        return out

    def __radd__(self, number):
        return self + number

    def __sub__(self, number):
        number = number if isinstance(number, Value) else Value(number)
        out = Value(self.data - number.data, '-', children=[self, number])
        return out

    def __rsub__(self, number):
        number = number if isinstance(number, Value) else Value(number)
        return number - self

    def __mul__(self, number):
        number = number if isinstance(number, Value) else Value(number)
        out = Value(self.data * number.data, '*', children=[self, number])
        return out

    def __rmul__(self, number):
        return self * number

    def __pow__(self, number):
        number = number if isinstance(number, Value) else Value(number)
        out = Value(self.data ** number.data, '**', children=[self, number])
        return out

    def __gt__(self, number):
        number = number if isinstance(number, Value) else Value(number)
        if self.data > number.data:
            return True
        return False

    def __lt__(self, number):
        number = number if isinstance(number, Value) else Value(number)
        if self.data < number.data:
            return True
        return False

    def tanh(self):
        out = Value(math.tanh(self.data), 'tanh', children=[self])
        return out

    def sigmoid(self):
        out = Value(1/(1+math.exp(-self.data)), 'sigmoid', children=[self])
        return out

    def relu(self):
        out = Value(max(0, self.data), 'relu', children=[self])
        return out

    def leaky_relu(self):
        val = self.data if self.data >= 0 else 0.01*self.data
        out = Value(val, 'leaky_relu', children=[self])
        return out

    def exp(self):
        out = Value(math.exp(self.data), 'exp', children=[self])
        return out

    def log(self):
        out = Value(math.log(self.data), 'log', children=[self])
        return out

    def abs(self):
        out = Value(abs(self.data), 'abs', children=[self])
        return out

    def make_topo(self):

        topo = []
        visited = set()
        stack = [self]

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build(child)
                topo.append(v)

        build(self)
        return topo

    def update(self):
        self.grad = 0
        match self.op:
            case '+':
                self.data = self.children[0].data + self.children[1].data
            case '-':
                self.data = self.children[0].data - self.children[1].data
            case '*':
                self.data = self.children[0].data * self.children[1].data
            case '**':
                self.data = self.children[0].data ** self.children[1].data
            case 'tanh':
                self.data = math.tanh(self.children[0].data)
            case 'sigmoid':
                self.data = 1/(1+math.exp(-self.children[0].data))
            case 'relu':
                self.data = max(0, self.children[0].data)
            case 'leaky_relu':
                data = self.children[0].data
                self.data = data if data >= 0 else 0.01 * data
            case 'exp':
                self.data = math.exp(self.children[0].data)
            case 'log':
                self.data = math.log(self.children[0].data)
            case 'abs':
                self.data = abs(self.children[0].data)

    # set the gradient of children
    def backprop(self, topo):
        """
        Calculate the gradients of the entire topology.
        Obviously the derivative of the output is 1, and then
        just propagate the gradient to all of the children
        """
        self.grad = 1

        for node in topo:
            match node.op:
                case '+':
                    node.children[0].grad += node.grad
                    node.children[1].grad += node.grad
                case '*':
                    node.children[0].grad += node.children[1].data * node.grad
                    node.children[1].grad += node.children[0].data * node.grad
                case '-':
                    node.children[0].grad += node.grad
                    node.children[1].grad -= node.grad
                case '**':
                    node.children[0].grad += node.children[1].data * (node.children[0].data ** (node.children[1].data -1)) * node.grad
                case 'tanh':
                    node.children[0].grad += (1 - node.data**2) * node.grad
                case 'sigmoid':
                    node.children[0].grad += (1 - node.data) * node.data * node.grad
                case 'relu':
                    node.children[0].grad += node.grad * (node.data > 0)
                case 'leaky_relu':
                    d = 0.01 if node.data < 0 else 1
                    node.children[0].grad += node.grad * d
                case 'exp':
                    node.children[0].grad += node.grad * node.data
                case 'log':
                    node.children[0].grad += node.grad * (1/node.children[0].data)
                case 'abs':
                    if node.children[0].data > 0:
                        d = 1
                    elif node.children[0].data < 0:
                        d = -1
                    else:
                        d = 0
                    node.children[0].grad += node.grad * d

    def __repr__(self):
        if not self.label:
            return f"data = {self.data}, gradient = {self.grad}, op = {self.op}"
        else:
            return f"label = {self.label}, data = {self.data}, gradient = {self.grad}, op = {self.op}"
