# Required Libraries
from pysat.solvers import Solver
from pysat.formula import CNF, IDPool
from pysat.card import CardEnc, EncType
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# ---------- Visualization Function ----------
def plot(queen_positions, n, gamma):
    """Plot the chessboard and mark queen positions."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1.5)

    # Color the board squares
    board_colors = np.zeros((n, n, 3))
    LIGHT_SQUARE = (240 / 255, 217 / 255, 181 / 255)
    DARK_SQUARE = (181 / 255, 136 / 255, 99 / 255)

    for r in range(n):
        for c in range(n):
            board_colors[r, c] = LIGHT_SQUARE if (r + c) % 2 == 0 else DARK_SQUARE

    ax.imshow(board_colors)

    # Place queens using Unicode symbol
    for row, col in [(divmod(pos, n)) for pos in queen_positions]:
        ax.text(col, row, "♛", fontsize=max(6, 200 // n), ha='center', va='center', color="black")

    plt.title(f"n = {n}, γ = {gamma}")
    plt.show()

# ---------- Problem Setup ----------
n, gamma = 13, 6  # Board size and max number of queens
N = n * n         # Total number of squares

# ID pool for variable management
vpool = IDPool(start_from=1)

# Define SAT variables
Q = lambda i: vpool.id(f"Q@{i}")  # True if square i contains a queen
R = lambda r: vpool.id(f"R@{r}")  # True if row r has no queens
C = lambda c: vpool.id(f"C@{c}")  # True if column c has no queens

# This approach can be extended to include diagonals and their complements, e.g., constraints like "at most gamma non-empty diagonals."
# However, in my experiments, the current version (using only row and column constraints via R and C) performs best.
# It significantly reduces runtime compared to the version without R and C definitions.
# Additionally, combining this formulation with symmetry breaking (e.g., using Satsuma) has shown further improvements in performance.

cnf = CNF()

# ---------- Domination Constraints ----------
# Each square must be dominated (either contain a queen or be threatened by one)
V = [Q(i) for i in range(N)]  # List of all queen variables
G = []  # Domination map

for i in range(N):
    r, c = divmod(i, n)
    Ns = [V[i]]

    # Add all squares in the same row
    Ns += V[r * n : r * n + n]

    # Add all squares in the same column
    Ns += V[c::n]

    # Add all squares on the same top-left to bottom-right diagonal
    Ns += [V[j] for j in range(N) if (j // n) - (j % n) == r - c]

    # Add all squares on the same top-right to bottom-left diagonal
    Ns += [V[j] for j in range(N) if (j // n) + (j % n) == r + c]

    Ns = list(set(Ns))  # Remove duplicates
    G.append(Ns)

    # At least one of the Ns must contain a queen
    cnf.append([Qx for Qx in Ns])

# ---------- Row Constraints ----------
# R(r) ↔ All squares in row r are empty (¬Q)
for r in range(n):
    row_vars = V[r * n : r * n + n]
    for q in row_vars:
        cnf.append([-R(r), -q])  # If row is empty, no square in it has a queen
    cnf.append([R(r)] + row_vars)  # If all squares are empty, R(r) is True

# ---------- Column Constraints ----------
# C(c) ↔ All squares in column c are empty (¬Q)
for c in range(n):
    col_vars = V[c::n]
    for q in col_vars:
        cnf.append([-C(c), -q])
    cnf.append([C(c)] + col_vars)

# ---------- Cardinality Constraints ----------
# At most γ queens on the board
cnf.extend(CardEnc.atmost(lits=V, vpool=vpool, bound=gamma, encoding=EncType.seqcounter))

# At least n - γ rows and columns must be completely empty
cnf.extend(CardEnc.atleast(lits=[R(r) for r in range(n)], vpool=vpool, bound=n - gamma, encoding=EncType.seqcounter))
cnf.extend(CardEnc.atleast(lits=[C(c) for c in range(n)], vpool=vpool, bound=n - gamma, encoding=EncType.seqcounter))

# ---------- SAT Solving ----------
start_time = datetime.now()
SAT = False

with Solver(name='Cadical195', bootstrap_with=cnf, use_timer=True) as solver:
    if solver.solve():
        model = solver.get_model()
        queens = [m - 1 for m in model if 0 < m <= N]  # Adjust to 0-based index
        SAT = True
    else:
        print("UNSAT")

    print(f"Solve time (internal): {solver.time_accum():.4f} seconds")

end_time = datetime.now()
elapsed_time = (end_time - start_time).total_seconds()

# ---------- Output ----------
if SAT:
    print("SAT")
    plot(queens, n, gamma)

print(f"Total runtime (wall-clock): {elapsed_time:.4f} seconds")
