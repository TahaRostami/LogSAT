from pysat.card import CardEnc, EncType
from pysat.formula import CNF
from pysat.solvers import Solver
import matplotlib.pyplot as plt
import numpy as np

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

def get_top_id(clauses):
    """ Given a list of clauses, it returns the maximum id
    :param clauses: a list of lists
    :return: maximum id
    """
    return max([max([abs(item) for item in c]) for c in clauses])

def queen_dom_to_SAT(n, gamma, enc_type_atmost=EncType.seqcounter, enc_type_atleast=EncType.seqcounter):
    """ Encodes the queen domination problem into the SAT problem
    :param n: the number of rows in a nxn chess board
    :param gamma: the size of the domination set
    :param enc_type: the type of cardinality encodings
    :return: corresponding SAT formula
    """

    if enc_type_atmost is None:
        raise Exception("enc_type_atmost must not be None")

    clauses = []
    # To ensure compatibility with SAT solvers, indices start from 1 and end at n^2
    V = [i + 1 for i in range(n * n)]
    top_id = V[-1]

    for i in range(len(V)):
        N = [V[i]]
        r, c = i // n, i % n
        # Squares in the same row
        N += V[r * n:r * n + n]
        # Squares in the same column
        N += V[c::n]
        # Squares in the same diagonal (from top left to bottom right)
        N += [V[j] for j in range(len(V)) if r - c == ((j // n) - (j % n))]
        # Squares in the same diagonal (from top right to bottom left)
        N += [V[j] for j in range(len(V)) if r + c == ((j // n) + (j % n))]
        N = list(set(N))
        clauses.append(N)

    clauses += CardEnc.atmost(lits=V, top_id=top_id, bound=gamma, encoding=enc_type_atmost).clauses

    if enc_type_atleast is not None:
        top_id = get_top_id(clauses)
        clauses += CardEnc.atleast(lits=V, top_id=top_id, bound=gamma, encoding=enc_type_atleast).clauses
        
    # OR
    #clauses += CardEnc.equals(lits=[-v for v in V], top_id=top_id, bound=((n*n)-gamma), encoding=enc_type_atmost).clauses
    
    return clauses

# Solve an Instance:
n = 8
gamma = 5
cnf = CNF(from_clauses=queen_dom_to_SAT(n, gamma))
SAT = False
with Solver(name='Cadical195', bootstrap_with=cnf) as solver:
    if solver.solve():
        model = solver.get_model()
        # Extract relevant values and adjust for zero-indexed positions
        queens = [m-1 for m in model if 0 < m <= n * n]
        SAT=True
    else:
        print("UNSAT")

if SAT:
    print("SAT")
    plot(queens, n, gamma)