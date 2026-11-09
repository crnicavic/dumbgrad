import requests
import os
import json
import random

from dumbgrad.nn import *
from dumbgrad.utils import *

if __name__ == "__main__":
    # if the file is already there dont download
    url = "https://huggingface.co/datasets/jerteh/SrpELTeC/resolve/main/SrpELTeC.jsonl?download=true"

    CHARACTERS = sorted(set("abcčćdđefghijklmnoprsštuvzž .,!?#"))
    mapping = create_mapping(CHARACTERS)

    dest_path = "./examples/datasets/srpeltec.jsonl"

    if not os.path.exists(dest_path):
        response = requests.get(url)
        f = open(dest_path, 'w')
        data = response.content.decode()
        f.write(data)
        f.close()
    else:
        f = open(dest_path, 'r')
        data = f.read()
        f.close()

    data = data.lower()
    lines = data.split("\n")
    novels= []
    for line in lines[:-1]:
        novel = json.loads(line)["text"]
        novel = novel.lower().replace("<s>", " ").replace("</s>", " ")
        novel = "".join(ch if ch in CHARACTERS else "#" for ch in novel)
        novels.append(novel)


    x = []
    for novel in novels:
        section_count = 10
        for _ in range(section_count):
            section_len = random.randint(50,100)
            section_start = random.randint(0, len(novel) - section_len)
            section_end = section_start + section_len
            section = novel[section_start:section_end]
            x.append(section)

    encoded = encode_labels(x, per_character=True, mapping=mapping)
    inputs = [e[:-1] for e in encoded]
    outputs = [e[1:] for e in encoded]

    input_count = len(mapping)
    output_count = input_count

    lstm_size = 50
    n = Network([
        Input(input_count),
        LSTM(lstm_size),
        Layer(output_count, activation="softmax")
    ])
    opt = Optimizer(lr=0.01)
    n.build(seed=0, optimizer=opt, loss="cross_entropy")
    n.train(inputs, outputs, epochs=50, n_workers=6)
    n.test(inputs, outputs, n_workers=6)

    dest_dir = "./examples/nets/"
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    dest_path = dest_dir + "srpeltec.json"
    n.save(dest_path)
