from dumbgrad.engine import Value
from dumbgrad.nn import *
from dumbgrad.utils import *

import os


def try_download(url, dest_path):
    import gzip
    import requests
    # if the file is already there dont download
    if os.path.exists(dest_path):
        return
    data = requests.get(url)
    stuff = gzip.decompress(data.content)
    f = open(dest_path, 'wb')
    f.write(stuff)

def load_idx3(path):
    f = open(path, 'rb')

    magic = int.from_bytes(f.read(4), byteorder='big')
    assert magic == 2051,"BAD FILE HEADER!"
    image_count = int.from_bytes(f.read(4), byteorder='big')
    image_w = int.from_bytes(f.read(4), byteorder='big')
    image_h = int.from_bytes(f.read(4), byteorder='big')
    images = []
    for i in range(image_count):
        image = [int.from_bytes(f.read(1)) for _ in range(image_w*image_h)]
        images.append(image)

    return images

def load_idx1(path):
    f = open(path, 'rb')

    magic = int.from_bytes(f.read(4), byteorder='big')
    assert magic == 2049, "BAD FILE HEADER!"

    label_count = int.from_bytes(f.read(4), byteorder='big')
    labels = []
    for i in range(label_count):
        label = int.from_bytes(f.read(1))
        labels.append(label)

    return labels

def draw_some_images(images):
    import matplotlib.pyplot as plt
    plt.subplot(2,2,1)
    for sim in range(1,5):
        plt.subplot(2,2,sim)
        im = []
        for i in range(28):
            im.append(images[sim][i*28:i*28+28])
        plt.imshow(im)
    plt.show()

if __name__ == "__main__":
    name_train_images =  "train-images-idx3-ubyte"
    name_train_labels =  "train-labels-idx1-ubyte"
    name_test_images =  "t10k-images-idx3-ubyte"
    name_test_labels =  "t10k-labels-idx1-ubyte"

    dest_dir = "./examples/datasets/fashion/"
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    for name in [name_train_images,
              name_train_labels,
              name_test_images,
              name_test_labels]:

        try_download(f"{base_url}{name}.gz", f"{dest_dir}{name}")

    train_images = load_idx3(f"{dest_dir}{name_train_images}")
    train_labels = load_idx1(f"{dest_dir}{name_train_labels}")
    test_images = load_idx3(f"{dest_dir}{name_test_images}")
    test_labels = load_idx1(f"{dest_dir}{name_test_labels}")

    num_classes = len(unique(test_labels))
    x_train = normalize(train_images)
    y_train = to_categorical(train_labels, num_classes)

    x_test = normalize(test_images)
    y_test = to_categorical(test_labels, num_classes)

    n = Network([
        Input(len(x_train[0])),
        Layer(30),
        Layer(30),
        Layer(num_classes, activation="softmax")
    ])
    reg = L2Regularization()
    n.build(seed=0, loss="cross_entropy", regularization=reg)
    n.train(x_train, y_train, batch_size=10, epochs=10, n_workers=6)
    n.test(x_test, y_test, n_workers=2)
