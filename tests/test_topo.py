from dumbgrad.engine import Value
import random

def test_topo_sanity():
    """
    A very simple test of topological sorting.
    Call make_topo and check if every node is
    added only after all of it's children are.

    In other words, call make_topo and check that
    each node in the resulting topology has all
    of it's children in the topology before it.
    The way this works is that it creates a set
    of visited nodes, and for each node, it checks
    if all of the children have been visited.
    """
    dims = 100
    x = [Value(random.random()) for _ in range(dims)]

    y = sum([xi**2 for xi in x])
    topo = y.make_topo()

    visited = set()
    for i, n in enumerate(topo):
        for c in n.children:
            assert c in visited
        visited.add(n)
