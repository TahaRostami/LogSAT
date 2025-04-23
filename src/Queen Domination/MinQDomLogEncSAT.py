from pysat.solvers import Solver
import matplotlib.pyplot as plt
import numpy as np
import math
from pysat.formula import *


"""
Log-Based SAT Encoding for the Queen Domination Problem

Overview:
---------
This script explores an alternative SAT encoding for the Queen Domination Problem using a logarithmic variable 
representation.
Rather than assigning one Boolean variable per board cell per queen (as in the conventional encoding), this method 
encodes each queen’s position using binary representation—significantly reducing the number of variables.

The script also explores how the inclusion of redundant constraints—while logically unnecessary—can guide the solver
and potentially reduce solving time.

Key Parameters:
---------------
- `n`     : Size of the chessboard (n×n)
- `gamma` : Number of queens used for domination

Boolean Variables:
------------------
- The position of each queen is encoded using ⌈log₂(n²)⌉ Boolean variables. 
    Therefore, total number of Boolean variables: gamma × ⌈log₂(n²)⌉.


Encoding Steps:
---------------
1. Bit Variable Assignment: For each of the `gamma` queens, assign ⌈log₂(n²)⌉ Boolean variables representing their 
                            position.
2. Range Restriction: Add clauses to ensure that each binary-encoded position corresponds to a 
                        valid square (i.e., in the range [0, n² - 1]).

3. Domination Constraints: Add clauses to guarantee that every square is dominated by at least one queen.

4. (Optional) Constraints: Add constraints to reduce redundancy in the search space (e.g., no two queens on the same
                           square).

Note:
-----
While this encoding is elegant and compact in terms of variable count, thereby reducing the potential search space,
it does not always result in improved performance for SAT solvers. The CDCL algorithms often face difficulties with
the bit-level abstraction, as it obscures higher-level semantic structures. As a result, the solver's efficiency can
be hindered, leading to longer solving times despite the reduced number of variables.

"""

def plot(queen_positions,n,gamma):
    # Visualization
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1.5)

    board_colors = np.zeros((n, n, 3))  # 3D array for RGB colors

    # Define board colors in normalized RGB format
    LIGHT_SQUARE = (240 / 255, 217 / 255, 181 / 255)  # Light Brown
    DARK_SQUARE = (181 / 255, 136 / 255, 99 / 255)  # Dark Brown
    for r in range(n):
        for c in range(n):
            board_colors[r, c] = LIGHT_SQUARE if (r + c) % 2 == 0 else DARK_SQUARE

    ax.imshow(board_colors)

    # Place queens
    for row, col in [(divmod(pos, n)) for pos in queen_positions]:
        ax.text(col, row, "♛", fontsize=max(6, 200 // n), ha='center', va='center', color="black")

    plt.title(f"n={n},gamma={gamma}")
    plt.show()

# ========== Problem Setup ==========

n,gamma = 8, 5
N=n*n
N_bits=math.ceil(math.log2(N))
solver=Solver(name='Cadical195',use_timer=True)

vname2id,id2vname={},{}

# Define binary variables for each queen
for qid in range(gamma):
    for b in range(N_bits):
        vname2id[f"Q_{qid}_bit_{b}"]=Atom(len(vname2id)+1)
        id2vname[Atom(len(id2vname)+1)]=f"Q_{qid}_bit_{b}"

print("Total Boolean Variables:", len(vname2id))

int_to_bin = [] # Precompute binary strings
for i in range(2 ** N_bits):
    int_to_bin.append("{0:0{1}b}".format(i, N_bits))

# Compute domination map G[i]: list of all squares that dominate cell i
V = [i for i in range(N)]
G=[]
for i in range(len(V)):
        Ns = [V[i]]
        r, c = i // n, i % n
        # Squares in the same row
        Ns += V[r * n:r * n + n]
        # Squares in the same column
        Ns += V[c::n]
        # Squares in the same diagonal (from top left to bottom right)
        Ns += [V[j] for j in range(len(V)) if r - c == ((j // n) - (j % n))]
        # Squares in the same diagonal (from top right to bottom left)
        Ns += [V[j] for j in range(len(V)) if r + c == ((j // n) + (j % n))]
        Ns = list(set(Ns))
        G.append(Ns)

# ========== Clause Construction ==========

# 1. Exclude invalid binary encodings (outside [0, N-1])
for idx in range(N, len(int_to_bin)):
    bits = int_to_bin[idx]
    for q in range(gamma):
        solver.append_formula(Or(*[Neg(vname2id[f'Q_{q}_bit_{i}']) if v=='1' else vname2id[f'Q_{q}_bit_{i}'] for i, v in enumerate(bits)]))

# 2. Domination constraint: Every cell must be dominated by at least one queen
for i in range(len(G)):
    Ns=G[i]
    ands=[]
    for q in range(gamma):
        for vidx in Ns:
            ands.append(And(*[Neg(vname2id[f'Q_{q}_bit_{i}']) if v=='0' else vname2id[f'Q_{q}_bit_{i}'] for i, v in enumerate(int_to_bin[vidx])]))
    solver.append_formula(Or(*ands))


# 3. Optional constraint: Prevent queens from occupying same square
for q1 in range(gamma):
    for q2 in range(q1+1,gamma):
        for idx in range(len(G)):
            solver.append_formula(Or(*([Neg(vname2id[f'Q_{q1}_bit_{i}']) if v == '1' else vname2id[f'Q_{q1}_bit_{i}'] for i, v in enumerate(int_to_bin[idx])]+[Neg(vname2id[f'Q_{q1}_bit_{i}']) if v == '1' else vname2id[f'Q_{q2}_bit_{i}'] for i, v in enumerate(int_to_bin[idx])])))


print("Encoding complete.")


if solver.solve():
    print("SAT")
    model = solver.get_model()
    vname2value={}
    for m in model:
        if Atom(abs(m)) in id2vname:
            vname2value[id2vname[Atom(abs(m))]]=1 if m>0 else 0

    queens=set()
    for q in range(gamma):
        queens.add(int(''.join([str(vname2value[f'Q_{q}_bit_{x}']) for x in range(N_bits)]), 2))

    print(f"Queens Placed: {len(queens)}")
    plot(queens, n, gamma)

else:
    print("UNSAT")

print("Solving Time:", solver.time_accum(), "seconds")
solver.delete()


"""
Performance Comparison
----------------------
Empirical timing results comparing log-based encoding vs conventional encoding with cardinality:

Log-Based Encoding (without optional clauses):
- n=10, gamma=4  → UNSAT in ~12s
- n=11, gamma=4  → UNSAT in ~18s
- n=12, gamma=5  → Timeout > 720s

Log-Based Encoding (with optional constraints):
- n=10, gamma=4  → UNSAT in ~7.5s
- n=11, gamma=4  → UNSAT in ~12.5s
- n=12, gamma=5  → Timeout > 720s

Conventional Encoding (explicit queen-square vars + cardinality):
- n=10, gamma=4  → UNSAT in ~9s
- n=11, gamma=4  → UNSAT in ~13s
- n=12, gamma=5  → UNSAT in ~227s
"""