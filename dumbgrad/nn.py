from dumbgrad.engine import Value
from dumbgrad.utils import *
import math
import random
import time
import multiprocessing as mp
import copy

def sum_of_squares(_y, _y_pred):
    y = flat_iter(_y)
    y_pred = flat_iter(_y_pred)
    diffs = [(y1 - y2)**2 for y1, y2 in zip(y, y_pred)]

    return value_sum(diffs)

def cross_entropy(_y, _y_pred):
    y = flat_iter(_y)
    y_pred = flat_iter(_y_pred)
    ent = []
    for y1, y2 in zip(y, y_pred):
        if y1 == 0:
            continue
        ent.append(-1 * y1 * y2.log())

    return value_sum(ent)

class Parameter(Value):
    __slots__ = ('m', 'v')
    def __init__(self, data, op=None, children=[], label=''):
        super().__init__(data, op, children, label)
        self.m = 0
        self.v = 0

class Optimizer:
    def __init__(self, omega1=0.9, omega2=0.99, lr=0.001, eps=1e-6):
        self.omega1 = omega1
        self.omega2 = omega2
        self.lr = lr
        self.eps = eps

    def __call__(self, p, t):
        p.m = self.omega1 * p.m + (1 - self.omega1) * p.grad
        p.v = self.omega2 * p.v + (1 - self.omega2) * p.grad**2

        m_hat = p.m / (1 - self.omega1 ** t)
        v_hat = p.v / (1 - self.omega2 ** t)

        p.data = p.data - self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

class Regularization():
    def __init__(self, alpha=0.01):
        self.alpha = alpha

class L1Regularization(Regularization):
    def __call__(self, weights):
        total = weights[0].abs()
        for w in weights[1:]:
            total += w.abs()
        return total * self.alpha

class L2Regularization(Regularization):
    def __call__(self, weights):
        total = weights[0] ** 2
        for w in weights[1:]:
            total += w ** 2
        return total * self.alpha

class NoRegularization(Regularization):
    def __call__(self, weights):
        return 0

class Neuron:
    def __init__(self, input_count, output_count, rng=None, activation="tanh"):
        limit = math.sqrt(6 / (input_count + output_count))
        if rng is None:
            self.w = [Parameter(random.uniform(-limit, limit), label='w') for _ in range(input_count)]
        else:
            self.w = [Parameter(rng.uniform(-limit, limit), label='w') for _ in range(input_count)]
        self.b = Parameter(0,label='b')

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
        """
        # using the value_sum function here is better avoided
        # but it's the same thing
        act = self.b
        for wi, xi in zip(self.w, x):
            act += wi * xi
        return self.activation(act)
        """
        return self.activation(Value.linear(self.w, x, self.b))

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, size, activation="tanh"):
        self.size = size
        self.activation = activation

    def __call__(self, x):
        out = [n(x) for n in self.neurons]

        # sum the activation of the layer to get softmax
        if self.activation == "softmax":
            total_act = value_sum(out)

            inv_total = total_act ** -1
            out = [o * inv_total for o in out]
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
        #
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

    def build(self,
              seed=None,
              loss="sum_of_squares",
              optimizer=None,
              regularization=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = None

        if loss == "sum_of_squares" or loss is None:
            self.loss = sum_of_squares
        elif loss == "cross_entropy":
            self.loss = cross_entropy

        if optimizer is None:
            self.optimizer = Optimizer()
        else:
            self.optimizer = optimizer

        if regularization is None:
            self.regularization = NoRegularization()
        else:
            self.regularization = regularization

        # dont build the first layer!
        for prev_layer, layer in zip(self.layers, self.layers[1:]):
            layer.build(prev_layer.size, rng)

        self.layers.pop(0)

    def train(self, inputs, outputs, batch_size=1, epochs=10, n_workers=1):
        assert len(inputs) == len(outputs), "Input and output size mismatch!"
        assert batch_size > 0 and batch_size <= len(outputs), "bad batch_size!"
        start_time = time.perf_counter()

        input_batches, output_batches = make_batches(inputs, outputs, batch_size)

        # don't create more processes than batches
        batch_count = math.floor(len(inputs)/batch_size)
        n_workers = n_workers if batch_count > n_workers else batch_count

        input_queues = [mp.Queue() for _ in range(n_workers)]
        output_queues = [mp.Queue() for _ in range(n_workers)]
        split_input_batches = array_split(input_batches, n_workers)
        split_output_batches = array_split(output_batches, n_workers)

        processes = [mp.Process(target=self.training_worker,
                                args=(split_input_batches[i],
                                      split_output_batches[i],
                                      output_queues[i],
                                      input_queues[i]
                                      ),
                                name=f"training_worker{i}"
                                )
                     for i in range(n_workers)]

        for p in processes:
            p.start()

        # cache parameters because self.parameters() is "slow"
        params = self.parameters()

        for t in range(1, epochs+1):
            for output_queue in output_queues:
                output_queue.put(params)

            # each message is a tuple of (loss, [gradients])
            messages = [input_queue.get() for input_queue in input_queues]

            losses, grads = map(list, zip(*messages))
            print(f"loss in epoch {t}: {sum(losses)}")
            # apply averaged gradients
            for p, g in zip(params, list(zip(*grads))):
                p.grad = sum(g)/batch_count

            for p in params:
                self.optimizer(p, t)

        for output_queue in output_queues:
            output_queue.put(None)

        for p in processes:
            p.join()

        training_time = time.perf_counter() - start_time
        print(f"training time on {len(outputs)} samples with {n_workers} workers: {training_time}s")

    def training_worker(self, inputs, outputs, input_queue, output_queue):
        placeholders_x = [[Value(col) for col in row] for row in inputs[0]]
        placeholders_y = [[Value(col) for col in row] for row in outputs[0]]
        y_pred = [self(x) for x in placeholders_x]
        loss = self.loss(placeholders_y, y_pred) + self.regularization(self.weights())
        topo = loss.make_topo()
        params = self.parameters()
        grads = [0 for _ in range(len(params))]

        while True:
            recv_params = input_queue.get()
            if recv_params is None:
                break

            for p, p_new in zip(params, recv_params):
                p.data = p_new.data
            grads[:] = [0 for _ in grads]

            for input_batch, output_batch in zip(inputs, outputs):
                update_placeholders(placeholders_x, input_batch)
                update_placeholders(placeholders_y, output_batch)
                loss.recompute(topo)
                loss.backprop(topo)
                for i in range(len(grads)):
                    grads[i] += params[i].grad
            output_queue.put((loss.data, grads))

    def test(self, inputs, outputs, n_workers=1):
        start_time = time.perf_counter()

        # don't create more processes than there are samples
        n_workers = n_workers if len(inputs) > n_workers else len(inputs)
        split_inputs = array_split(inputs, n_workers)
        split_outputs = array_split(outputs, n_workers)
        queue = mp.Queue()
        processes = [mp.Process(target=self.testing_worker,
                                args=(split_inputs[i],
                                      split_outputs[i],
                                      queue),
                                name=f"training_worker{i}"
                                )
                     for i in range(n_workers)]

        for p in processes:
            p.start()

        # combine the per class correct guesses into one dict
        total_correct = 0
        for _ in range(n_workers):
            worker_correct = queue.get()
            total_correct += worker_correct

        for p in processes:
            p.join()

        accuracy = total_correct / len(outputs)
        print(f"total accuracy: {accuracy}")
        testing_time = time.perf_counter() - start_time
        print(f"testing time on {len(outputs)} samples with {n_workers} workers: {testing_time}s")
        return accuracy


    def testing_worker(self, inputs, outputs, queue):
        """
        When calling the network object, it creates a prediction
        for the input. The prediction is a list of value objects
        The approach is to make a dummy node with all of those
        objects as children. By calling make topo, a single
        recompute call will do the entire prediction again.

        By updating the inputs (via placeholders), the entire
        and then calling recompute, it is possible to do very
        fast predictions whilst using very little memory
        """
        placeholders_x = [Value(i) for i in inputs[0]]
        pred = self(placeholders_x)
        dummy = Value(0, children=pred)
        topo = dummy.make_topo()
        correct_count = 0
        for i in range(len(inputs)):
            update_placeholders(placeholders_x, inputs[i])
            dummy.recompute(topo)
            correct_count += int(argmax([p.data for p in pred]) == argmax(outputs[i]))

        queue.put(correct_count)
