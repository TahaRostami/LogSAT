import networkx as nx
import matplotlib.pyplot as plt
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver

# Graph coloring problem with at most k colors
# Given a graph G, can we color the vertices using at most k distinct colors
# such that no two adjacent vertices share the same color?

k = 3
G = [
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
]

# Another example
# G = [
#     [0, 1, 1, 1, 0, 0, 0, 0],
#     [1, 0, 1, 1, 0, 0, 0, 0],
#     [1, 1, 0, 1, 0, 0, 0, 0],
#     [1, 1, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1, 1, 0],
#     [0, 0, 0, 0, 1, 0, 1, 0],
#     [0, 0, 0, 0, 1, 1, 0, 1],
#     [0, 0, 0, 0, 0, 0, 1, 0]
# ]





colors = [i + 1 for i in range(k)]

# List to hold the clauses for the SAT solver
clauses = []
# Initialize the variable pool, starting from ID 1
vpool = IDPool(start_from=1)

# Whether vertex i-th has color c-th
V = lambda i, c: vpool.id(f'V{i}@{c}')

# Each vertex must have exactly one color
for i in range(1, len(G) + 1):
    # Each vertex must have at least one color
    clauses.append([V(i, c) for c in colors])

    # Each vertex must have at most one color
    for c1 in colors:
        for c2 in colors:
            if c1 != c2:
                clauses.append([-V(i, c1), -V(i, c2)])

# Two adjacent vertices must have different colors
for i in range(1, len(G) + 1):
    for j in range(1, len(G) + 1):
        if i != j and G[i - 1][j - 1] != 0:
            for c in colors:
                clauses.append([-V(i, c), -V(j, c)])

cnf = CNF(from_clauses=clauses)


# Solve the SAT problem
with Solver(name='Cadical153', bootstrap_with=cnf) as solver:
    if solver.solve():
        print("SAT")
        model = solver.get_model()

        # Extract coloring from model
        coloring = {}

        for i in range(1, len(G) + 1):
            for c in colors:
                if V(i, c) in model:
                    coloring[i - 1] = c
                    
        # Visualize the graph and coloring
        graph = nx.Graph()
        for i in range(len(G)):
            graph.add_node(i)
        for i in range(len(G)):
            for j in range(len(G)):
                if G[i][j] != 0:
                    graph.add_edge(i, j)

        node_colors = [coloring[node] for node in graph.nodes()]

        plt.figure(figsize=(8, 6))
        nx.draw(graph, with_labels=True, node_color=node_colors, cmap=plt.cm.rainbow, edge_color='gray')
        plt.title("Graph Coloring (SAT Solution)")
        plt.show()
    else:
        print("UNSAT")
