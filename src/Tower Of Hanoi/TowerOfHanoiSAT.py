import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver

def solve_instance(N=3, R=3, T=8):
    """
    Solves the Tower of Hanoi problem for a given number of disks (N), rods (R),
    and time steps (T) using a SAT solver approach.

    This function generates a set of logical clauses that represent the constraints
    of the Tower of Hanoi puzzle, such as disk movement, rod placement, and the
    goal state. It then uses a SAT solver to find a sequence of moves that
    satisfies these constraints, if a solution exists.

    Parameters:
    - N (int): The number of disks in the puzzle.
    - R (int): The number of rods in the puzzle (typically 3).
    - T (int): The number of time steps allowed for the solution.

    Returns:
    - True if a solution is found, False otherwise.

    The function also visualizes the solution using matplotlib, animating the movement
    of disks between rods over time.
    """
    vpool = IDPool(start_from=1)

    disks = list(range(1, N + 1))
    rods = list(range(1, R + 1))

    # True if disk d placed on rod r at time t
    Disk = lambda d, r, t: vpool.id(f'Disk@{d}@{r}@{t}')
    # True if disk d is moved from rod src to rod des at time t
    Move = lambda disk,r_src, r_des, t: vpool.id(f'Move@{disk}@{r_src}@{r_des}@{t}')

    # Assigns IDs
    for d in disks:
       for r in rods:
           for t in range(T):
               Disk(d,r,t)

    for d in disks:
        for src in rods:
            for des in rods:
                if src!=des:
                    for t in range(T):
                        Move(d,src,des,t)



    clauses = []

    # Disk movement constraints
    for t in range(T - 1):
        for d in disks:
            for src_rod in rods:
                for des_rod in rods:
                    if src_rod != des_rod:
                        # If move occurs, update the disk's position
                        clauses.append([-Move(d, src_rod, des_rod, t), Disk(d, src_rod, t)])
                        clauses.append([-Move(d, src_rod, des_rod, t), -Disk(d,des_rod,t)])
                        clauses.append([-Move(d, src_rod, des_rod, t), -Disk(d, src_rod, t + 1)])
                        clauses.append([-Move(d, src_rod, des_rod, t), Disk(d, des_rod, t + 1)])

    # Assuming the move is valid, disks other than the moving disk will remain in the same position as before
    for t in range(T-1):
        for moving_disk in disks:
            for src_rod in rods:
                for des_rod in rods:
                    if src_rod!=des_rod:
                        for other_disk in disks:
                            if moving_disk!=other_disk:
                                for r in rods:
                                    # If moving disk from src_rod to des_rod at time t, then for any other disk,
                                    # the disk remains in the same rod at time t+1 (if it was on rod r at time t)

                                    #Move(moving_disk,src_rod,des_rod,t)=>(Disk(other_disk,r,t)=>Disk(other_disk,r,t+1))
                                    clauses.append([-Move(moving_disk,src_rod,des_rod,t),-Disk(other_disk,r,t),Disk(other_disk,r,t+1)])

                                    #Move(moving_disk,src_rod,des_rod,t)=>(-Disk(other_disk, r, t) => -Disk(other_disk, r, t + 1))
                                    clauses.append([-Move(moving_disk,src_rod,des_rod,t),Disk(other_disk,r,t),-Disk(other_disk, r, t + 1)])

    # Assuming disk d is in rod r at time t, moving it is valid if it is the top disk in its rod,
    # and it will be the top disk in the destination rod.
    # A disk is considered the top disk if it has the lowest number among other disks in the
    # same rod (i.e., no disk with a larger number is above it).
    for t in range(T-1):
        for moving_disk in disks:
            for src_rod in rods:
                for des_rod in rods:
                    if src_rod!=des_rod:
                        # Get all disks smaller than the moving disk
                        potentially_smaller_disks=[smaller_disk for smaller_disk in disks if smaller_disk<moving_disk]

                        # If we want to move the disk from src_rod to des_rod at time t,
                        # we can do so if there are no smaller disks remaining on the source rod at time t
                        # (i.e., the moving disk must be the top disk on src_rod).
                        #Move(moving_disk, src_rod, des_rod, t) => -Disk(psd,  src_rod, t)
                        for psd in potentially_smaller_disks:
                            clauses.append([-Move(moving_disk, src_rod, des_rod, t),-Disk(psd,  src_rod, t)])

                        # Similarly, we can move the disk to des_rod at time t if there are no smaller disks
                        # remaining on the destination rod at that time step
                        # (i.e., the moving disk must become the top disk on des_rod).
                        #Move(moving_disk, src_rod, des_rod, t) => -Disk(psd, des_rod, t)
                        for psd in potentially_smaller_disks:
                            clauses.append([-Move(moving_disk, src_rod, des_rod, t),-Disk(psd, des_rod, t)])

    # At least one move per time step
    for t in range(T-1):
        clauses.append([Move(d,src,des,t) for d in disks for src in rods for des in rods if src!=des])
    # # At most one move per time step
    for t in range(T-1):
        for d1 in disks:
            for src1 in rods:
                for des1 in rods:
                    if src1 != des1:
                        for d2 in disks:
                            for src2 in rods:
                                for des2 in rods:
                                    if src2 != des2 and (d1, src1, des1) < (d2, src2, des2):
                                        clauses.append([-Move(d1, src1, des1, t), -Move(d2, src2, des2, t)])

    # No move occurs at the final time step (T-1) because at that time,
    # we expect all disks to be in their final goal positions (for the SAT instances).
    for d in disks:
        for src in rods:
            for des in rods:
                if src != des:
                    clauses.append([-Move(d, src, des, T-1)])

    # Initial State: All disks start on the first rod
    for d in disks:
        clauses.append([Disk(d, rods[0], 0)])
        for r in rods:
            if r != rods[0]:
                clauses.append([-Disk(d, r, 0)])

    # Goal State: All disks must be on the last rod at time T-1
    for d in disks:
        clauses.append([Disk(d, rods[-1], T-1)])
        for r in rods:
            if r != rods[-1]:
                clauses.append([-Disk(d, r, T-1)])


    # Solve using SAT solver
    sat=False
    cnf = CNF(from_clauses=clauses)
    with Solver(name='cadical153', bootstrap_with=cnf) as solver:
        if solver.solve():
            model = solver.get_model()
            print(T-1)
            # Extract disk positions at each step
            state_by_time = {t: {r: [] for r in rods} for t in range(T)}
            move_by_time={}

            for m in model:
                if m > 0:
                    var_name = vpool.id2obj[m]
                    if var_name.startswith('Disk'):
                        _, d, r, t = var_name.split('@')
                        state_by_time[int(t)][int(r)].append(int(d))
                    elif var_name.startswith('Move'):
                        _,disk, r_src, r_des, t=var_name.split('@')
                        move_by_time[int(t)]=var_name

            # Define disk colors (using a colormap for disks)
            disk_colors = {d: plt.cm.viridis(d / N) for d in range(1, N + 1)}

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.set_xlim(-0.5, R - 0.5)
            ax.set_ylim(0, N + 1)
            ax.set_xticks(range(R))
            ax.set_xticklabels([f'Rod {r}' for r in rods])
            ax.set_xlabel('Rods')
            ax.set_ylabel('Disk Position')
            ax.set_title('Tower of Hanoi Simulation')


            def update(frame):
                ax.clear()

                ax.set_facecolor('black')

                # Redraw bars for each rod (represented as a vertical bar)
                for r in range(R):
                    ax.bar(r, N + 0.8, width=0.009, bottom=0, color='white', zorder=1)

                # Redraw axis elements
                ax.set_xlim(-0.5, R - 0.5)
                ax.set_ylim(0, N + 1)
                ax.set_xticks(range(R))
                ax.set_xticklabels([f'Rod {r}' for r in rods])
                ax.set_title(f'Time Step {frame}', color='white')

                # Plot the disks as rectangles placed on top of each other
                for r in rods:
                    for i, d in enumerate(sorted(state_by_time[frame][r], reverse=True)):
                        # Use ax.bar to create rectangular disks
                        ax.bar(r - 1, 1, width=d/5, bottom=i, color=disk_colors[d], zorder=2)

                ax.set_xlabel('Rods', color='white')
                ax.set_ylabel('Disk Position', color='white')

            ani = animation.FuncAnimation(fig, update, frames=T, interval=500, repeat=False)

            plt.tight_layout()
            plt.show()

            sat= True
    return sat


# Find the minimum number of moves needed
for t in range(32, 100):
    if solve_instance(N=5, R=3, T=t):
        break




