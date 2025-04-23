from pysat.formula import CNF, IDPool
from pysat.solvers import Solver

# List to hold the clauses for the SAT solver
clauses = []
# Initialize the variable pool, starting from ID 1
vpool = IDPool(start_from=1)

# Board size and configuration options
n = 8  # Board size
enb_closed_tour = [True,False][1]  # Enables closed tour if Knight must finish near the starting square
starting_square = (3, 3)  # Initialize starting square (1-based index [1, n*n]) or None for random start

# Define the variable for knight's position at (i, j) at time t
N = lambda i, j, t: vpool.id(f'S{i}@{j}@{t}')

# Generate variables for knight's positions at each time step
for i in range(1, n + 1):
    for j in range(1, n + 1):
        for t in range(1, n * n + 1):
            N(i, j, t)

# Ensure each square is visited exactly once
for i in range(1, n + 1):
    for j in range(1, n + 1):
        for t1 in range(1, n * n + 1):
            for t2 in range(1, n * n + 1):
                if t1 != t2:
                    clauses.append([-N(i, j, t1), -N(i, j, t2)])

# Constraint: Knight must be in exactly one square (i, j) at each time step
# At least one square must be occupied at each time step
for t in range(1, n * n + 1):
    clauses.append([N(i, j, t) for i in range(1, n + 1) for j in range(1, n + 1)])

# At most one square per time step
for t in range(1, n * n + 1):
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            for iprim in range(1, n + 1):
                for jprim in range(1, n + 1):
                    if (i, j) != (iprim, jprim):
                        clauses.append([-N(i, j, t), -N(iprim, jprim, t)])

# Define knight's possible moves
potential_moves = [(-2, -1), (-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (-1, -2), (1, -2)]
all_squares = {(i, j) for i in range(1, n + 1) for j in range(1, n + 1)}

# Enforce the knight's movement constraints based on allowed moves
for i in range(1, n + 1):
    for j in range(1, n + 1):
        # Get valid neighboring squares based on knight's movement
        neighbors = {(i + dx, j + dy) for dx, dy in potential_moves if 1 <= i + dx <= n and 1 <= j + dy <= n}
        not_neighbors = all_squares - neighbors

        # Prevent knight from appearing in non-neighboring squares in the next step
        for iprim, jprim in not_neighbors:
            for t in range(1, n * n):
                clauses.append([-N(i, j, t), -N(iprim, jprim, t + 1)])

        # If closed tour is enabled, the knight must finish in the neighborhood of the starting square
        if enb_closed_tour:
            clauses.append([-N(i, j, 1)] + [N(iprim, jprim, n * n) for iprim, jprim in neighbors])

# Initial condition: Knight starts at time 1
if starting_square is not None:
    clauses.append([N(starting_square[0], starting_square[1], 1)])

print('Encoded!')

cnf = CNF(from_clauses=clauses)
SAT = False
# Solve the SAT problem
with Solver(name='Cadical195', bootstrap_with=cnf) as solver:
    if solver.solve():
        print("SAT")

        board = [[-1] * n for _ in range(n)]
        model = solver.get_model()
        for m in model:
            if 0 < m <= len(vpool.id2obj):
                i, j, t = map(int, vpool.id2obj[m][1:].split('@'))
                board[i - 1][j - 1] = t

        for row in board:
            print(row)
        SAT = True

    else:
        print("UNSAT")


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

# Update function for the animation, displaying the knight's movements
def update(frame):
    move = frame + 1
    if move in positions:
        row, col = positions[move]
        correct_x = col + 0.5
        correct_y = (n - row - 1) + 0.5

        knight_marker.set_position((correct_x, correct_y))

        ax.add_patch(patches.Rectangle((col, n - row - 1), 1, 1, facecolor="yellow", alpha=0.5))

        if (col, row) not in visited_squares:
            ax.text(col + 0.5, n - row - 0.5, str(move), ha="center", va="center", fontsize=12, color="black", zorder=4)

        for prev_move in range(1, move):
            prev_row, prev_col = positions[prev_move]
            if (prev_col, prev_row) not in visited_squares:
                ax.add_patch(patches.Rectangle((prev_col, n - prev_row - 1), 1, 1, facecolor="lightblue", alpha=0.5))
                ax.text(prev_col + 0.5, n - prev_row - 0.5, str(prev_move), ha="center", va="center", fontsize=12,
                        color="black", zorder=4)
                visited_squares[(prev_col, prev_row)] = True

# If a solution was found, prepare for animation
if SAT:

    positions = {board[i][j]: (i, j) for i in range(n) for j in range(n)}

    start_row, start_col = positions[1]

    fig, ax = plt.subplots(figsize=(6, 6))

    for i in range(n):
        for j in range(n):
            color = "white" if (i + j) % 2 == 0 else "gray"
            ax.add_patch(patches.Rectangle((j, n - i - 1), 1, 1, facecolor=color))

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    knight_symbol = '♞'

    correct_x = start_col + 0.5
    correct_y = (n - start_row - 1) + 0.5
    knight_marker = ax.text(correct_x, correct_y, knight_symbol, fontsize=36, ha="center", va="center", color="black",
                            zorder=3)

    visited_squares = {}

    ani = animation.FuncAnimation(fig, update, frames=n * n, interval=500, repeat=False)

    plt.show()
