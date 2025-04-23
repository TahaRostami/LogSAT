import numpy as np
import matplotlib.pyplot as plt
from pysat.solvers import Solver
from pysat.card import CardEnc, EncType

def get_candidate_squares(i, n):
    """Returns all squares attacked by a queen at index i on an n x n chessboard."""
    candidate_squares = []
    row, col = divmod(i, n)

    # Main diagonal (\)
    for d in range(-n, n):
        if 0 <= row + d < n and 0 <= col + d < n:
            candidate_squares.append((row + d, col + d))

    # Anti-diagonal (/)
    for d in range(-n, n):
        if 0 <= row + d < n and 0 <= col - d < n:
            candidate_squares.append((row + d, col - d))

    # Same row
    for c in range(n):
        candidate_squares.append((row, c))

    # Same column
    for r in range(n):
        candidate_squares.append((r, col))

    # Convert to 1-based indexing
    return list(set((r * n + c) + 1 for r, c in candidate_squares if (r, c) != (row, col)))

def naive_enc_nqueens(n):
    clauses = []

    # No two queens should attack each other
    for i in range(n * n):
        vars = get_candidate_squares(i, n)
        for j in range(len(vars)):
            if i + 1 != vars[j]:
                clauses.append([-((i + 1)), -vars[j]])

    ## Exactly N queens in total
    clauses += CardEnc.equals(lits=[i+1 for i in range(n*n)], bound=n, encoding=EncType.seqcounter).clauses
    return clauses

def enc_nqueens(n):
    clauses = []

    #  No two queens should attack each other
    for i in range(n * n):
        vars = get_candidate_squares(i, n)
        for j in range(len(vars)):
            if i+1 != vars[j]:
                clauses.append([-((i + 1)), -vars[j]])

    # Exactly one queen per row and column
    for i in range(n):
        row_vars = [(i * n + col) + 1 for col in range(n)]
        col_vars = [(row * n + i) + 1 for row in range(n)]

        clauses += CardEnc.equals(lits=row_vars, bound=1, encoding=EncType.pairwise).clauses
        clauses += CardEnc.equals(lits=col_vars, bound=1, encoding=EncType.pairwise).clauses

    return clauses


def plot(queen_positions,n):
    # Visualization
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1.5)

    # Draw the chessboard with custom colors
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

    plt.title(f"n={n}")
    plt.show()

n =8
encoder=[enc_nqueens,naive_enc_nqueens][0]
clauses=encoder(n)
# Solve using SAT solver
with Solver(name='Cadical195', bootstrap_with=clauses) as solver:
    if solver.solve():
        model = solver.get_model()

        print("SAT:")
        # Extract queen positions from model (1-based indices) such that
        # Convert to 0-based index while excluding auxiliary variables
        queen_positions = [i - 1 for i in model if 0 < i <= n*n]
        board=[[0]*n for _ in range(n)]
        for x in [n*n]+model:
            if x>0 and x<=n*n:
                i,j=divmod(x-1,n)
                board[i][j]=1
        for r in board:
            print(r)
        plot(queen_positions,n)

    else:
        print("UNSAT")
