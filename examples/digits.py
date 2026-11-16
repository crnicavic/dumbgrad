from dumbgrad.engine import Value
from dumbgrad.nn import *
from dumbgrad.utils import *
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import os

if __name__ == "__main__":
    dataset = load_digits()
    x, y = dataset.data.tolist(), dataset.target.tolist()
    normalize(x)
    num_classes = len(unique(y))
    y = to_categorical(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=0)

    n = Network([
        Input(len(x[0])),
        Layer(30),
        Layer(30),
        Layer(num_classes, activation="softmax")
    ])
    reg = L2Regularization()
    optimizer = Optimizer(lr=0.01)
    n.build(seed=0, loss="cross_entropy", regularization=reg, optimizer=optimizer)
    n.train(x_train, y_train, batch_size=419, epochs=20, n_workers=2)
    n.test(x_test, y_test, n_workers=1)

    dest_dir = "./examples/nets/"
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    dest_path = dest_dir + "digits.json"
    n.save(dest_path)
