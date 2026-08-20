from dumbgrad.engine import Value
from dumbgrad.utils import *
import math
import random
import time
import multiprocessing as mp
import copy

def sum_of_squares(_y, _y_pred):
    y = flatten(_y)
    y_pred = flatten(_y_pred)
    diff = [(y1 - y2)**2 for y1, y2 in zip(y, y_pred)]
    loss = sum(diff)
    return loss

def cross_entropy(_y, _y_pred):
    y = flatten(_y)
    y_pred = flatten(_y_pred)
    ent = []
    for y1, y2 in zip(y, y_pred):
        if y1 == 0:
            continue
        ent.append(-1 * y1 * y2.log())
    loss = sum(ent)
    return loss

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

    def train(self, inputs, outputs, batch_size=1, epochs=10, n_jobs=1):
        assert len(inputs) == len(outputs), "Input and output size mismatch!"
        assert batch_size > 0 and batch_size <= len(outputs), "bad batch_size!"
        start_time = time.perf_counter()

        batches = make_batches(inputs, outputs, batch_size)

        # if there are more batches then specified workers, limit
        # so that only the necessary amount of processes are created
        n_jobs = n_jobs if len(batches) > n_jobs else len(batches)

        # NOTE: for now eacn batch gets it's own process
        input_queues = [mp.Queue() for _ in range(n_jobs)]
        output_queues = [mp.Queue() for _ in range(n_jobs)]
        split_batches = array_split(batches, n_jobs)
        processes = [mp.Process(target=self.worker,
                                args=(split_batches[i],
                                      output_queues[i],
                                      input_queues[i]
                                      )
                                )
                     for i in range(n_jobs)]

        for p in processes:
            p.start()
        # cache to avoid triple list comp every loop
        params = self.parameters()
        for t in range(1, epochs+1):
            for output_queue in output_queues:
                output_queue.put(copy.deepcopy(params))

            inbox = [input_queue.get() for input_queue in input_queues]

            losses, grads = map(list, zip(*inbox))
            print(f"loss in epoch {t}: {sum(losses)}")
            # apply gradients
            for p, g in zip(params, list(zip(*grads))):
                p.grad = sum(g)/len(batches)

            for p in params:
                self.optimizer(p, t)

        end_time = time.perf_counter()
        print(f"training time on {len(outputs)} samples with {n_jobs} workers: {end_time - start_time}s")

        for output_queue in output_queues:
            output_queue.put(None)

    def worker(self, batches, input_queue, output_queue):
        def update_placeholders(placeholders, new_values):
            for placeholder, new_val in zip(placeholders, new_values):
                placeholder.data = new_val

        batch_in, batch_out = batches[0]
        placeholders_x = [[Value(col) for col in row] for row in batch_in]
        placeholders_y = [[Value(col) for col in row] for row in batch_out]
        y_pred = [self(x) for x in placeholders_x]
        loss = self.loss(placeholders_y, y_pred) + self.regularization(self.weights())
        topo = loss.make_topo()

        placeholders_x = flatten(placeholders_x)
        placeholders_y = flatten(placeholders_y)

        batches = [(flatten(bi), flatten(bo)) for bi, bo in batches]
        params = self.parameters()
        grads = [0 for _ in range(len(params))]
        while True: 
            recv_params = input_queue.get()
            if recv_params is None:
                break
            for p, p_new in zip(params, recv_params):
                p.data = p_new.data
            grads[:] = [0 for _ in grads]
            for batch in batches:
                batch_in, batch_out = batch
                update_placeholders(placeholders_x, batch_in)
                update_placeholders(placeholders_y, batch_out)
                loss.recompute(topo)
                loss.backprop(topo)
                for i in range(len(grads)):
                    grads[i] += params[i].grad
            output_queue.put((loss.data, grads))

    def test(self, inputs, outputs):
        y_pred = [self(x) for x in inputs]
        correct_count = 0
        for pred, output in zip(y_pred, outputs):
            if argmax(pred) == argmax(output):
                correct_count += 1

        accuracy = correct_count / len(outputs)
        print(f"accuracy on {len(outputs)} test samples: {accuracy}")
        out_uniq = unique(from_categorical(outputs))
        pred_uniq = unique(from_categorical(y_pred))
        print("Model classification stats:")
        print(list(pred_uniq.keys()))
        for k in out_uniq:
            if k not in pred_uniq:
                pred_uniq[k] = 0
            print(f"\tclass {k} expected: {out_uniq[k]}, got: {pred_uniq[k]}")

        return accuracy
