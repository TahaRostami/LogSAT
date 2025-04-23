import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def update(frame_num, img, grid):
    new_grid = grid.copy()
    rows, cols = grid.shape

    for i in range(rows):
        for j in range(cols):
            total = 0
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        total += grid[ni, nj]

            if grid[i, j] == 1:
                if total < 2 or total > 3:
                    new_grid[i, j] = 0
            else:
                if total == 3:
                    new_grid[i, j] = 1

    img.set_data(new_grid)
    grid[:] = new_grid[:]
    return img


def run_game_of_life(initial_grid, interval=200):
    fig, ax = plt.subplots()

    img = ax.imshow(initial_grid, interpolation='nearest', cmap='binary',
                    extent=(0, initial_grid.shape[1], 0, initial_grid.shape[0]))

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xticks(np.arange(0, initial_grid.shape[1]+1, 1), minor=True)
    ax.set_yticks(np.arange(0, initial_grid.shape[0]+1, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)

    ani = animation.FuncAnimation(fig, update, fargs=(img, initial_grid),
                                  frames=100, interval=interval, blit=False)
    plt.show()


# Initial configuration
initial_grid=np.array([
[1,1,0,1,0,0,0,1,1,0,0,1],
[0,1,0,1,0,0,0,1,1,0,0,1],
[0,1,1,0,1,1,0,1,1,1,1,0],
[1,0,0,1,0,0,0,0,0,0,1,0],
[0,1,0,0,1,1,0,0,1,0,1,0],
[0,1,0,1,1,1,0,0,0,0,1,0],
[1,1,0,1,1,0,0,1,1,0,1,1],
[0,0,0,0,0,0,1,1,0,1,0,0],
[0,1,1,0,1,0,0,1,0,0,1,1],
[1,0,0,1,1,0,1,0,0,0,1,1],
[0,0,0,1,0,0,1,1,0,1,1,1],
[1,1,0,0,1,1,1,0,1,0,0,1],
])

run_game_of_life(initial_grid)
