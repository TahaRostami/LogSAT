import matplotlib.pyplot as plt
import numpy as np
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver

def visualize_latin_square(solution, n=4):
    grid = np.zeros((n, n), dtype=int)
    for var in solution:
        if var > 0:
            name = vpool.obj(var)
            i, j, k = map(int, name[1:].split("@"))
            grid[i, j] = k + 1

    colors = plt.cm.get_cmap("tab10", n)

    fig, ax = plt.subplots(figsize=(7, 7))

    for i in range(n):
        for j in range(n):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=colors(grid[i, j] - 1)))
            ax.text(j + 0.5, i + 0.5, str(grid[i, j]), va='center', ha='center',
                    fontsize=20, weight='bold', color='black')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)

    for i in range(n + 1):
        ax.axhline(i, color='black', linewidth=2)
        ax.axvline(i, color='black', linewidth=2)

    plt.gca().invert_yaxis()
    plt.show()


vpool = IDPool(start_from=1)
n = 7
C = lambda i, j, k: vpool.id(f'C{i}@{j}@{k}')
clauses = []

# Each cell has at least one value
for i in range(n):
    for j in range(n):
        clauses.append([C(i, j, k) for k in range(n)])

# Each cell has at most one value
for i in range(n):
    for j in range(n):
        for k1 in range(n):
            for k2 in range(k1 + 1, n):
                clauses.append([-C(i, j, k1), -C(i, j, k2)])

# Each value appears once per row and column
for k in range(n):
    for e in range(n):
        # Each value appears exactly once in each row
        clauses.append([C(e, j, k) for j in range(n)])
        for j1 in range(n):
            for j2 in range(j1 + 1, n):
                clauses.append([-C(e, j1, k), -C(e, j2, k)])

        # Each value appears exactly once in each column
        clauses.append([C(j, e, k) for j in range(n)])
        for j1 in range(n):
            for j2 in range(j1 + 1, n):
                clauses.append([-C(j1, e, k), -C(j2, e, k)])

cnf = CNF(from_clauses=clauses)
with Solver(name='cadical153', bootstrap_with=cnf) as solver:
    if solver.solve():
        print("SAT")
        model = solver.get_model()
        visualize_latin_square(model, n)
    else:
        print("UNSAT")
