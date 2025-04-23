import math
import numpy as np
import matplotlib.pyplot as plt
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver


vpool = IDPool(start_from=1)
S = lambda i, j, k: vpool.id(f'S{i}@{j}@{k}')

for i in range(1,10):
    for j in range(1,10):
        for k in range(1,10):
            S(i,j,k)

clauses = []

# Each square must be filled with a digit between 1 and 9
for i in range(1, 10):
    for j in range(1, 10):
        clauses.append([S(i, j, k) for k in range(1, 10)])

# And that the same digit cannot appear twice in the same row, column, or block
for k in range(1, 10):
    for i in range(1, 10):
        for j in range(1, 10):
            for iprim in range(1, 10):
                for jprim in range(1, 10):
                    if (i, j) != (iprim, jprim):
                        same_row = i == iprim
                        same_col = j == jprim
                        same_block = (math.ceil(i / 3) == math.ceil(iprim / 3)) and (
                            math.ceil(j / 3) == math.ceil(jprim / 3))

                        if same_row or same_col or same_block:
                            clauses.append([-S(i, j, k), -S(iprim, jprim, k)])

# Given Sudoku Instance
#Collins, N.: World’s hardest sudoku: can you crack it? https://www.telegraph.co.
#uk/news/science/science-news/9359579/Worlds-hardest-sudoku-can-you-crack-it.html (2012)
instance = [
    (1, 1, 8), (2, 3, 3), (2, 4, 6),
    (3, 2, 7), (3, 5, 9), (3, 7, 2),
    (4, 2, 5), (4, 6, 7), (5, 5, 4),
    (5, 6, 5), (5, 7, 7), (6, 4, 1),
    (6, 8, 3), (7, 3, 1), (7, 8, 6),
    (7, 9, 8), (8, 3, 8), (8, 4, 5),
    (8, 8, 1), (9, 2, 9), (9, 7, 4)
]

# Convert sudoku instance to unit clauses
instance_clauses = [[S(i, j, k)] for (i, j, k) in instance]

# Solve Sudoku
cnf = CNF(from_clauses=clauses + instance_clauses)

solution = np.zeros((9, 9), dtype=int)
original_positions = {(i, j): k for (i, j, k) in instance}

SAT=False
with Solver(bootstrap_with=cnf) as solver:
    if solver.solve():
        model = solver.get_model()
        SAT=True
    else:
        print("UNSAT")

if SAT:
    # Extract solution from model
    for var in model:
        if var > 0:
            try:
                parts = vpool.obj(var).split('@')
                i, j, k = int(parts[0][1:]), int(parts[1]), int(parts[2])
                solution[i - 1][j - 1] = k
            except:
                continue

    print("\nSolved Sudoku:")
    for i in range(9):
        for j in range(9):
            num = solution[i, j]
            if (i + 1, j + 1) in original_positions:
                print(f"({num})", end=" ")  # Prefilled values in parentheses
            else:
                print(f" {num} ", end=" ")
        print()

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks(np.arange(0, 9, 1))
    ax.set_yticks(np.arange(0, 9, 1))
    ax.set_xticks(np.arange(-0.5, 9, 3), minor=True)
    ax.set_yticks(np.arange(-0.5, 9, 3), minor=True)
    ax.grid(which="minor", color="black", linewidth=2)
    ax.grid(which="major", color="gray", linestyle="dotted")

    for i in range(9):
        for j in range(9):
            num = solution[i, j]
            if (i + 1, j + 1) in original_positions:
                ax.text(j, i, str(num), ha='center', va='center', fontsize=16, color="black",
                        fontweight="bold")  # Prefilled values
            else:
                ax.text(j, i, str(num), ha='center', va='center', fontsize=16, color="blue")  # Solver-filled values

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.invert_yaxis()
    plt.show()

