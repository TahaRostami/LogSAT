from pysat.solvers import Solver
from pysat.formula import CNF, IDPool

# Initialize variable pool for unique SAT variables
vpool = IDPool(start_from=1)

# Parameters
graphical_viz=True
T = 41  # Number of time steps in the plan
K = 4   # Number of knights (2 white: 0,1 and 2 black: 2,3)

# Define valid board squares
squares = {
    (0,1),
    (1,1), (1,2),
    (2,1), (2,2), (2,3),
    (3,0), (3,1), (3,2), (3,3)
}

def get_neighbors(i, j):
    """Returns valid knight moves from position (i, j) within the board constraints."""
    moves = {(-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (-1, -2), (1, -2), (-2, -1)}
    return [(i + di, j + dj) for di, dj in moves if (i + di, j + dj) in squares]

# Define SAT variables
N = lambda i, j, k, t: vpool.id(f"N@{i}@{j}@{k}@{t}")  # Knight k is at (i, j) at time t
Z = lambda k, t: vpool.id(f"Z@{k}@{t}")  # Knight k is chosen to move at time t

# Clause collection
clauses = []

# Constraint: Each knight must be in exactly one square per time step
for k in range(K):
    for t in range(T):
        clauses.append([N(x, y, k, t) for x, y in squares]) # At least one position
        for (x1, y1) in squares:
            for (x2, y2) in squares:
                if (x1, y1) < (x2, y2):  # Avoid duplicate pairs
                    clauses.append([-N(x1, y1, k, t), -N(x2, y2, k, t)]) # No two positions at once

# Constraint: Legal knight moves or staying in place
for t in range(T - 1):
    for k in range(K):
        for x, y in squares:
            clauses.append([-N(x, y, k, t), N(x, y, k, t + 1)] + [N(x1, y1, k, t + 1) for x1, y1 in get_neighbors(x, y)])

for k in range(K):
    for t in range(T - 1):
        for x, y in squares:
            # If knight k moves, it must vacate its previous position
            # (Z(k,t) and N(x,y,k,t)) => -N(x,y,k,t+1)
            clauses.append([-Z(k, t),-N(x,y,k,t), -N(x, y, k, t+1)])
            # If knight k does not move, it stays in the same position
            # (-Z(k,t) and N(x,y,k,t)) => N(x,y,k,t+1)
            clauses.append([Z(k, t),-N(x,y,k,t), N(x, y, k, t+1)])

            for x1,y1 in squares:
                if (x,y)!=(x1,y1):
                    for k1 in range(K):
                        if k!=k1:
                            # If knight k moves to (x1, y1), no other knight can be there
                            clauses.append([-N(x,y,k,t),-N(x1,y1,k,t+1),-N(x1,y1,k1,t)])


# Constraint: Exactly one knight moves per timestep
for t in range(T - 2):
    clauses.append([Z(k, t) for k in range(K)])
    for k1 in range(K):
        for k2 in range(k1 + 1, K):
            clauses.append([-Z(k1, t), -Z(k2, t)])

for k in range(K):
    clauses.append([-Z(k, T-1)]) # No movement on the last step

# Initial positions of knights
initial_positions = [(0, 1, 0), (2, 2, 1), (3, 0, 2), (3, 2, 3)]
for x, y, k in initial_positions:
    clauses.append([N(x, y, k, 0)])

# Goal positions for knights
clauses.append([N(0,1,2,T-1),N(0,1,3,T-1)])
clauses.append([N(2,2,2,T-1),N(2,2,3,T-1)])
clauses.append([N(3,0,0,T-1),N(3,0,1,T-1)])
clauses.append([N(3,2,0,T-1),N(3,2,1,T-1)])

# Solve the CNF formula
cnf = CNF(from_clauses=clauses)
solution = None

with Solver(name='Cadical195', bootstrap_with=cnf) as solver:
    if solver.solve():
        print("SAT")
        solution = solver.get_model()
    else:
        print("UNSAT")

if solution is not None:
    knight_symbols = ['\u2658', '\u2658', '\u265E', '\u265E']  # Two white knights, two black knights
    result = [[None for _ in range(K)] for _ in range(T)]
    chosen_knight=[None for _ in range(T)]
    for idx in solution:
        if idx > 0 and idx in vpool.id2obj:
            if vpool.id2obj[idx].startswith("N"):
                _, i, j, k, t = vpool.id2obj[idx].split("@")
                result[int(t)][int(k)] = (int(i), int(j))
            else:
                _, k, t = vpool.id2obj[idx].split("@")
                chosen_knight[int(t)]=int(k)

    for i,res in enumerate(result):
        print(chosen_knight[i], res)

    if graphical_viz:
        def viz():
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            from matplotlib.patches import Rectangle

            # Define board colors
            light_color = "#F0D9B5"  # Light brown (like a chessboard)
            dark_color = "#B58863"  # Dark brown

            # Define figure
            fig, ax = plt.subplots(figsize=(6, 6))

            # Define the set of valid squares (rotated)
            valid_squares_rotated = [(y, -x) for x, y in squares]  # Rotating by 90 degrees

            # Find new board limits
            xmin, xmax = min(x for x, y in valid_squares_rotated), max(x for x, y in valid_squares_rotated)
            ymin, ymax = min(y for x, y in valid_squares_rotated), max(y for x, y in valid_squares_rotated)

            # Set up board limits
            ax.set_xlim(xmin - 0.5, xmax + 0.5)
            ax.set_ylim(ymin - 0.5, ymax + 0.5)

            # Hide axis
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_frame_on(False)

            # Draw the valid board squares in a chessboard pattern (rotated)
            for x, y in valid_squares_rotated:
                color = light_color if (x + y) % 2 == 0 else dark_color
                ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, color=color, ec="black", lw=1.5))

            # Knight symbols
            knight_symbols = ['♘', '♘', '♞', '♞']
            knight_texts = []

            # Initialize text elements at their correct starting positions
            for k in range(K):
                start_pos = result[0][k]  # Get the first position of each knight
                x, y = start_pos
                knight_text = ax.text(y, -x, knight_symbols[k], fontsize=45, ha="center", va="center",
                                      fontweight="bold")  # Corrected
                knight_texts.append(knight_text)

            # Function to update knight positions (rotated)
            def update(frame):
                for k in range(K):
                    pos = result[frame][k]
                    if pos:
                        x, y = pos
                        knight_texts[k].set_position((y, -x))  # Apply the same rotation to the knights

            # Create animation
            ani = animation.FuncAnimation(fig, update, frames=T, interval=500, repeat=False)
            plt.show()
        viz()





