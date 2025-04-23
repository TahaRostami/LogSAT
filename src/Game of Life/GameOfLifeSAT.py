"""
Problem Definition: Given a finite non-wrapping grid of size n×n and a target configuration to be reached in T
timesteps, assuming the rules of the Game of Life are applied, determine if there exists an initial configuration that
will evolve into the target. If such a configuration exists, find it; otherwise, declare it unsolvable.

Rules (from Wikipedia):
At each step in time, the following transitions occur:
- Any live cell with fewer than two live neighbours dies, as if by underpopulation.
- Any live cell with two or three live neighbours lives on to the next generation.
- Any live cell with more than three live neighbours dies, as if by overpopulation.
- Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.

SAT solvers are highly efficient, but they often lack expressive power. In scenarios where formulating a problem purely
in Boolean logic is either infeasible or not straightforward, we might prefer to use other types of solvers.

Just for demonstration purposes, here, I use Z3, an SMT (Satisfiability Modulo Theories) solver. SMT solvers extend SAT
solvers by allowing reasoning over richer theories such as arithmetic, arrays, bit-vectors, and strings, offering
significantly more expressive power than pure SAT.

However, this expressive power comes at a cost—SMT solvers are generally not as efficient as SAT solvers for large-scale
Boolean problems.

That said, this script demonstrates how SMT solvers like Z3 can be used to solve this particular problem.
"""


from z3 import *
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# Timesteps
T=5
# Grid size
n=12



rows, cols = n, n
target_grid = np.zeros((rows, cols), dtype=int)

# Plot setup: Black (1) = alive, White (0) = dead
fig, ax = plt.subplots()
img = ax.imshow(1 - target_grid, cmap='gray', vmin=0, vmax=1)  # 1 - value: black for 1

ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

ax.set_title("Click to toggle cells (Black = Live, White = Dead). Close when done.")

# On click: toggle cell
def onclick(event):
    if event.inaxes == ax:
        r, c = int(event.ydata), int(event.xdata)
        if 0 <= r < rows and 0 <= c < cols:
            target_grid[r][c] ^= 1  # Toggle 0 <-> 1
            img.set_data(1 - target_grid)  # Invert for display: 1 -> black
            fig.canvas.draw_idle()

fig.canvas.mpl_connect('button_press_event', onclick)
print("Click cells to toggle between live (black) and dead (white). Close when done.")
plt.show()

goal_grid = target_grid.copy()
print("Target configuration selected:")
print(goal_grid)

# === ENCODING ===
# C(t, r, c): True if the cell at row r, column c is alive at time t; otherwise, False (dead)
board = [[[Bool(f"C@{t}@{r}@{c}") for c in range(cols)] for r in range(rows)] for t in range(T+1)]
s = Solver()

def apply_life_rules(t):
    for r in range(rows):
        for c in range(cols):
            neighbors = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbors.append(board[t][nr][nc])

            live_neighbors = Sum([If(n, 1, 0) for n in neighbors])
            cell_now = board[t][r][c]
            cell_next = board[t+1][r][c]

            s.add(
                If(cell_now, # The cell is currently alive
                   If(Or(live_neighbors < 2, live_neighbors > 3),
                      cell_next == False, # Dies by underpopulation or overpopulation
                      cell_next == True),  # Lives on (2 or 3 neighbors)
                   If(live_neighbors == 3, # The cell is currently dead
                      cell_next == True, # Becomes alive by reproduction
                      cell_next == False) # Remains dead
                )
            )

# Apply Rules
for t in range(T):
    apply_life_rules(t)

# Goal State
for r in range(rows):
    for c in range(cols):
        s.add(board[T][r][c] == BoolVal(bool(goal_grid[r][c])))

# Solve
if s.check() == sat:
    m = s.model()
    generations = []
    for t in range(T+1):
        grid = np.zeros((rows, cols), dtype=int)
        for r in range(rows):
            for c in range(cols):
                grid[r][c] = 1 if is_true(m.evaluate(board[t][r][c])) else 0
        generations.append(grid)

    init_state = generations[0]

    print("\nInitial configuration:")
    print('[')
    for row in init_state:
        print('['+','.join([str(r) for r in row])+'],')
    print(']')

    diff_first = (generations[1] != generations[0])
    diff_last = (generations[T] != generations[T-1])

    # === PLOTTING ===
    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(1, 5)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(init_state, cmap='Greys', interpolation='none')
    ax1.set_title("Initial (t=0)")
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax_anim = fig.add_subplot(gs[0, 1:4])
    img = ax_anim.imshow(generations[0], cmap='Greys', interpolation='none')
    ax_anim.set_title("Evolution")
    ax_anim.set_xticks([])
    ax_anim.set_yticks([])

    ax2 = fig.add_subplot(gs[0, 4])
    ax2.imshow(goal_grid, cmap='Greys', interpolation='none')
    ax2.set_title(f"Target (t={T})")
    ax2.set_xticks([])
    ax2.set_yticks([])

    annotation_text = ax_anim.text(0.02, -0.3, "", transform=ax_anim.transAxes, fontsize=10)


    def update(frame):
        img.set_data(generations[frame])
        ax_anim.set_title(f"Step {frame}")

        if frame == 1:
            text = "First Move:\n" + ', '.join([f"({r},{c})" for r, c in np.argwhere(diff_first)])
            annotation_text.set_text(text)
        elif frame == T:
            text = "Last Move:\n" + ', '.join([f"({r},{c})" for r, c in np.argwhere(diff_last)])
            annotation_text.set_text(text)
        else:
            annotation_text.set_text("")
        return [img, annotation_text]


    ani = animation.FuncAnimation(fig, update, frames=T + 1, interval=800)

    plt.tight_layout()
    plt.show()

else:
    print(" X No initial config found that leads to the target in given steps.")
