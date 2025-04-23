from pysat.formula import CNF, IDPool
from pysat.solvers import Solver

max_t = 27

start_state = [7, 2, 4,
               5, 0, 6,
               8, 3, 1]

# goal_state = [7, 2, 4,
#               5, 6, 1,
#               8, 0, 3]

goal_state = [0, 1, 2,
              3, 4, 5,
              6, 7, 8]

neighbors = [[1, 3],
             [0, 2, 4],
             [1, 5],
             [0, 4, 6],
             [3, 1, 5, 7],
             [4, 2, 8],
             [3, 7],
             [6, 4, 8],
             [5, 7]]

clauses = []
vpool = IDPool(start_from=1)

# Variable encoding: whether cell at position p at time t has value k
C = lambda p, t, k: vpool.id(f'C{p}@{t}@{k}')

# Encode initial state
for p in range(9):
    clauses.append([C(p, 0, start_state[p])])

# Encode goal state
for p in range(9):
    clauses.append([C(p, max_t - 1, goal_state[p])])

# Each cell has exactly one value at each time step
for t in range(max_t):
    for p in range(9):
        # At least one value
        clauses.append([C(p, t, k) for k in range(9)])

        # At most one value (pairwise)
        for k1 in range(9):
            for k2 in range(k1 + 1, 9):
                clauses.append([-C(p, t, k1), -C(p, t, k2)])

# Each value must be in exactly one position at each time step
for t in range(max_t):
    for k in range(9):
        # At least one position
        clauses.append([C(p, t, k) for p in range(9)])

        # At most one position (pairwise)
        for p1 in range(9):
            for p2 in range(p1 + 1, 9):
                clauses.append([-C(p1, t, k), -C(p2, t, k)])

# Movement constraints (valid blank tile moves)
# If a cell p contains the blank tile (0) at time t, the blank must have come from a neighboring cell at time t - 1
for t in range(1, max_t):
    for p in range(9):
        # If position p has blank at time t, it must come from a neighbor
        clauses.append([-C(p, t, 0)] + [C(n, t - 1, 0) for n in neighbors[p]])

# If a non-blank value k is at position p at time t, then either:
# It was already there at time t-1 (no move)
# The blank is in a neighboring position at time t (valid swap)
for t in range(1, max_t):
    for p in range(9):
        for k in range(1,9):
            clauses.append([-C(p, t, k),C(p, t-1, k)] + [C(n, t, 0) for n in neighbors[p]])

# A and B swapped and nothing else
for t in range(1, max_t):
    for k in range(1,9):
        for a in range(9):
            for b in range(9):
                if a!=b:
                    clauses.append([-C(b, t, k),C(b, t-1, k),-C(a, t, 0),C(a, t-1, 0),C(b, t-1, 0)])
                    clauses.append([-C(b, t, k), C(b, t - 1, k), -C(a, t, 0), C(a, t - 1, 0), C(a, t - 1, k)])

print('Encoded!')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def visualize_solution(states):
    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])

    # Draw the grid
    def draw_grid(state):
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])

        grid = np.array(state).reshape(3, 3)

        for i in range(3):
            for j in range(3):
                value = grid[i, j]
                if value == 0:
                    color = "white"
                else:
                    color = "skyblue"
                ax.add_patch(plt.Rectangle((j, 2 - i), 1, 1, edgecolor='black', facecolor=color))
                if value != 0:
                    ax.text(j + 0.5, 2 - i + 0.5, str(value),
                            ha='center', va='center', fontsize=24, color='black')

        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        ax.set_aspect('equal')

    # Animate the solution
    ani = animation.FuncAnimation(fig, draw_grid, frames=states, interval=700, repeat=False)
    plt.show()

cnf = CNF(from_clauses=clauses)

# Solve the puzzle
with Solver(name='g3', bootstrap_with=cnf) as solver:
    if solver.solve():
        print("SAT")

        model = solver.get_model()
        states = [[0] * 9 for _ in range(max_t)]

        # Decode the solution
        for m in model:
            if m > 0:
                p, t, k = vpool.id2obj[m][1:].split('@')
                p, t, k = int(p), int(t), int(k)
                states[t][p] = k

        # Print the solution
        for t in range(max_t):
            print(f"\nTime {t}:")
            for i in range(3):
                print(states[t][i * 3:i * 3 + 3])

        visualize_solution(states)

    else:
        print("UNSAT")
