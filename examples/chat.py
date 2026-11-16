from dumbgrad.nn import *
from dumbgrad.utils import *
import os
import random

def sanitize(string):
    return "".join([ch if ch in CHARACTERS else "#" for ch in string.lower()])


CHARACTERS = sorted(set("abcčćdđefghijklmnoprsštuvzž .,!?#"))
mapping = create_mapping(CHARACTERS)

net_path = "./examples/nets/srpeltec.json"
if not os.path.exists(net_path):
    raise ValueError(f"{net_path} doesn't exist!")
n = Network.load(net_path)

prompt = sanitize(input("> "))
encoded = encode_labels(prompt, mapping=mapping)

reset = True
pred = None
# send the prompt to the model
for char in encoded:
    pred = n(char, reset=reset)
    reset = False

reply = ""
REPLY_LEN = 150
# mapping is char -> index, this gives me index -> char
reverse_mapping = {v: k for k, v in mapping.items()}
for _ in range(REPLY_LEN):
    pred = n(pred)
    reply += reverse_mapping[argmax(pred)]
    # avoid loops by picking the next character according to probabilities
    choice = random.choices(range(len(pred)), weights=[p.data for p in pred])[0]
    pred = [int(i == choice) for i in range(len(pred))]

print(reply)
