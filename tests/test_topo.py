from dumbgrad.engine import Value
import random

dims = 4
x = [Value(random.random()) for _ in range(dims)]

y = sum([xi**2 for xi in x])
