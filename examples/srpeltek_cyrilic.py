import requests
import os
import json
import random

from dumbgrad.nn import *
from dumbgrad.utils import *

if __name__ == "__main__":
    # if the file is already there dont download
    url = "https://huggingface.co/datasets/jerteh/SrpELTeC/resolve/main/SrpELTeC-plus.jsonl?download=true"

    dest_path = "./examples/datasets/srpeltec.jsonl"
    if not os.path.exists(dest_path):
        response = requests.get(url)
        f = open(dest_path, 'w')
        data = response.content.decode()
        data = data.replace("<s>", "")
        data = data.replace("</s>", "")
        f.write(data)
    else:
        f = open(dest_path, 'r')
        data = f.read()

    lines = data.split("\n")
    novels= []
    for part in data.split("\n")[:-2]:
        novels.append(json.loads(part))

    x = []
    # for each novel take 20 sections of a random size
    for novel in novels:
        section_count = 20
        for _ in range(section_count):
            section_start = random.randint(0, len(novel["text"]))
            section_len = random.randint(50,100)
            section_end = section_start + section_len
            section = novel["text"][section_start:section_end]
            x.append(section)

    encoded = encode_labels(x, per_character=True)
    inputs = [e[:-1] for e in encoded]
    outputs = [e[1:] for e in encoded]

    uniq = unique(x, per_character=True)
    input_count = len(uniq)
    output_count = input_count

    lstm_size = 50
    n = Network([
        Input(input_count),
        LSTM(lstm_size),
        Layer(output_count)
    ])
    opt = Optimizer(lr=0.01)
    n.build(seed=0, optimizer=opt)
    n.train(inputs, outputs, epochs=50, n_workers=4)
