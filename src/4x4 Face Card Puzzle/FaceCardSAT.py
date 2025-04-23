# Solving the 4x4 Face Card Puzzle using SAT
#
# Puzzle Description:
# - From a standard 52-card deck, only the aces and face cards (Jack, Queen, King) of each suit (Clubs, Diamonds,
#   Hearts, Spades) are used.
# - The objective is to fill a 4×4 grid such that each row, each column, and both major diagonals contain exactly
#   one card of each suit and one card of each rank.
#
# Puzzle Source: https://puzzlewocky.com/parlor-games/4x4-face-card-puzzle/

from pysat.solvers import Solver
from pysat.formula import *

ranks = ["Ace", "Jack", "Queen", "King"]
suits = ["♣", "♦", "♥", "♠"]
n = len(ranks)

solver = Solver(name='cadical153')
vpool = IDPool(start_from=1)

# Variables:
# R(i, j, r): Card at position (i, j) has rank r
# S(i, j, c): Card at position (i, j) has suit c
R = lambda i, j, r: vpool.id(f"Rank@{i}@{j}@{r}")
S = lambda i, j, c: vpool.id(f"Suit@{i}@{j}@{c}")

# Constraint Type 1: Ensuring each row and column forms a Latin square independently for ranks and suits

# Each cell must have exactly one rank and one suit
for i in range(n):
    for j in range(n):
        solver.add_clause([R(i, j, r) for r in range(n)])
        solver.add_clause([S(i, j, c) for c in range(n)])

# No duplicate ranks or suits within a cell
for i in range(n):
    for j in range(n):
        for k1 in range(n):
            for k2 in range(k1 + 1, n):
                solver.add_clause([-R(i, j, k1), -R(i, j, k2)])
                solver.add_clause([-S(i, j, k1), -S(i, j, k2)])

# Each rank and suit must appear exactly once in each row and column
for k in range(n):
    for e in range(n):
        # Each rank appears exactly once per row and column
        solver.add_clause([R(e, j, k) for j in range(n)])
        solver.add_clause([S(e, j, k) for j in range(n)])
        for j1 in range(n):
            for j2 in range(j1 + 1, n):
                solver.add_clause([-R(e, j1, k), -R(e, j2, k)])
                solver.add_clause([-S(e, j1, k), -S(e, j2, k)])

        solver.add_clause([R(j, e, k) for j in range(n)])
        solver.add_clause([S(j, e, k) for j in range(n)])
        for j1 in range(n):
            for j2 in range(j1 + 1, n):
                solver.add_clause([-R(j1, e, k), -R(j2, e, k)])
                solver.add_clause([-S(j1, e, k), -S(j2, e, k)])

# Constraint Type 2: Major diagonals must also be unique in rank and suit
main_diagonal = [(i, i) for i in range(n)]
anti_diagonal = [(i, n - 1 - i) for i in range(n)]

def apply_diagonal_constraints(diagonal):
    solver.add_clause([R(i, j, r) for i, j in diagonal for r in range(n)])
    solver.add_clause([S(i, j, c) for i, j in diagonal for c in range(n)])
    for (i1, j1) in diagonal:
        for (i2, j2) in diagonal:
            if (i1, j1) != (i2, j2):
                for k in range(n):
                    solver.add_clause([-R(i1, j1, k), -R(i2, j2, k)])
                    solver.add_clause([-S(i1, j1, k), -S(i2, j2, k)])

apply_diagonal_constraints(main_diagonal)
apply_diagonal_constraints(anti_diagonal)

# Constraint Type 3: Ensuring each (rank, suit) combination appears once in the grid
# This constraint is not naturally in CNF. Fortunately, PySAT offers facilities for handling non-clausal formulas
# and efficiently converting them into CNF by introducing auxiliary variables and using existing approaches such
# as Tseitin transformation.

for r in range(n):
    Rs = [Atom(R(i, j, r)) for i in range(n) for j in range(n)]
    for c in range(n):
        Cs = [Atom(S(i, j, c)) for i in range(n) for j in range(n)]
        f = Or(*[And(Rs[x], Cs[x]) for x in range(len(Rs))])
        # There is no need to call f.clausify() for CNF conversion; PySAT handles it automatically.
        solver.append_formula([cxx for cxx in f])

# Solving the puzzle and displaying the solution if one exists
if solver.solve():
    print("SAT")
    grid = [[[None, None] for _ in range(n)] for _ in range(n)]
    model = solver.get_model()
    for m in model:
        if m > 0 and m in vpool.id2obj:
            lbl, i, j, k = vpool.id2obj[m].split("@")
            grid[int(i)][int(j)][0 if lbl == "Rank" else 1] = ranks[int(k)] if lbl == "Rank" else suits[int(k)]
    for row in grid:
        for rank, suit in row:
            print(f"({rank[0]},{suit})", end="   ")
        print()
else:
    print("UNSAT")

solver.delete()
