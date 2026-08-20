import numpy as np
import math

class Value:
    __slots__ = ('data', 'grad', 'children', 'op', 'label')
    def __init__(self, data, op=None, children=[], label=''):
        self.data = data
        self.grad = 0 # what is the derrivative of the output by this variable
        self.children = children
        self.op = op
        self.label = label

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
        """
        Function that topologically sorts all of the nodes
        used to build the Value object. In other words,
        in order to add a node to the topology, all of it's
        children need to be added into it first.

        The way it works is that it "simulates recursion"

        A stack consisting of entries that contain the current
        value node, and the index of the child that is to be
        inserted into the stack. Another way of looking at it is
        the number of children that are in the stack.

        The class is just that, a container for a value object and
        the child index.

        The loop works as follows:
        - Get the last entry of the stack
        - Check if all of the children are in the STACK
            - if not - add the first of the children to
            the stack and increment how many children are in the stack
        """
        class stack_entry:
            def __init__(self, node):
                self.node = node
                self.i = 0

        topo = []
        visited = {self}
        stack = [stack_entry(self)]
        while stack:
            node, i = stack[-1].node, stack[-1].i
            if i < len(node.children):
                stack[-1].i += 1
                child = node.children[i]
                if child not in visited:
                    visited.add(child)
                    stack.append(stack_entry(child))
            else:
                topo.append(node)
                stack.pop()

        return topo

    def recompute(self, topo):
        """
        Iterate through a topology and recalculate
        the values of the nodes.

        This is useful if any of the children of a
        node would be changed.

        Used mainly in the training method of the
        Network class to avoid having to make a new
        graph after updating the parameters.
        """
        for node in topo:
            # NOTE: this has to be done, because if not,
            # backprop will keep accumulating gradients
            # which will explode at some point!
            node.grad = 0
            match node.op:
                case '+':
                    node.data = node.children[0].data + node.children[1].data
                case '-':
                    node.data = node.children[0].data - node.children[1].data
                case '*':
                    node.data = node.children[0].data * node.children[1].data
                case '**':
                    node.data = node.children[0].data ** node.children[1].data
                case 'tanh':
                    node.data = math.tanh(node.children[0].data)
                case 'sigmoid':
                    node.data = 1/(1+math.exp(-node.children[0].data))
                case 'relu':
                    node.data = max(0, node.children[0].data)
                case 'leaky_relu':
                    data = node.children[0].data
                    node.data = data if data >= 0 else 0.01 * data
                case 'exp':
                    node.data = math.exp(node.children[0].data)
                case 'log':
                    node.data = math.log(node.children[0].data)
                case 'abs':
                    node.data = abs(node.children[0].data)

    def backprop(self, topo):
        """
        Calculate the gradients of the entire topology.
        Obviously the derivative of the output is 1, and then
        just propagate the gradient to all of the children
        """
        self.grad = 1

        for node in reversed(topo):
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

