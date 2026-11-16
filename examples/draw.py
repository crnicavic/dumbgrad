from dumbgrad.nn import *
from dumbgrad.utils import *
import tkinter as tk
import os

ELEM_DEFAULT_COLOR = "#000000"
ELEM_PRESS_COLOR = "#FFFFFF"
ELEM_W = 4
ELEM_H = 2

GRID_W = 8
GRID_H = 8
grid = [0 for _ in range(GRID_W * GRID_H)]

def on_click(row, col, button):
    # set grid state and input state
    i = row * GRID_W + col
    grid[i] = grid[i] ^ 1
    inputs[i].data = grid[i]

    if grid[i]:
        button.config(bg=ELEM_PRESS_COLOR)
    else:
        button.config(bg=ELEM_DEFAULT_COLOR)

    # call recompute for number recognition
    dummy.recompute(topo)

    text_box.delete("1.0", tk.END)
    text_box.insert(tk.END, f"{argmax(pred)}")

root = tk.Tk()
root.title("draw")

# load net
net_path = "./examples/nets/digits.json"
if not os.path.exists(net_path):
    raise ValueError(f"{net_path} doesn't exist!")
n = Network.load(net_path)

# initialize the input placeholders for speed
inputs = [Value(g) for g in grid]
pred = n(inputs)
dummy = Value(0, children=pred)
topo = dummy.make_topo()

# create grid of buttons for digit input
for r in range(GRID_W):
    for c in range(GRID_H):
        button = tk.Button(
            root,
            width=ELEM_W,
            height=ELEM_H,
            bg="#000000"
        )
        # pass lambda to create unique callbacks for that grid position
        button.config(
            command=lambda row=r, col=c, b=button: on_click(row, col, b)
        )

        button.grid(row=r, column=c)

# create textbox
text_box = tk.Text(root, width=ELEM_W*GRID_W, height=2)
text_box.grid(row=GRID_H, column=0, columnspan=GRID_W)

# freedom button
button_quit = tk.Button(
    root,
    text="Quit",
    command=exit,
)
button_quit.grid(row=GRID_H+1, column=GRID_W//2 - 1, columnspan=2)

root.mainloop()
