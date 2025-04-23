from pysat.formula import CNF, IDPool
from pysat.solvers import Solver
from pysat.card import *
from pysat.pb import PBEnc

# The code works but may contain some redundancies or require revisions.

# Define the number of time steps
T=10

# Initialize variable pool
vpool = IDPool(start_from=1)
# True if missionary i-th in is position p in {left,boat,right} in time t
M = lambda i, p, t: vpool.id(f'Missionary@{i}@{p}@{t}')
# True if cannibal i-th in is position p in {left,boat,right} in time t
C = lambda i, p, t: vpool.id(f'Cannibal@{i}@{p}@{t}')
# Boat is in left bank if true and in right bank if it is False
B= lambda t:vpool.id(f'Boat@{t}')


clauses=[]
idx=[1,2,3]# Index for missionaries and cannibals
positions=['left','boat','right']# Possible positions

# Assign IDs to all possible variable states
for t in range(T):
    B(t)
    for i in idx:
        for p in positions:
            M(i,p,t)
            C(i,p,t)

# Retrieve the highest assigned variable ID so far
top_id=vpool.top

# Define the boat's movement constraints
for t in range(T):
    if t%2==0:# Boat is on the left at even time steps
        clauses.append([B(t)])
    else: # Boat is on the right at odd time steps
        clauses.append([-B(t)])


# Initial conditions (all entities start on the left bank)
clauses.append([-M(1,'right',0)])
clauses.append([-M(2,'right',0)])
clauses.append([-M(3,'right',0)])
clauses.append([-C(1,'right',0)])
clauses.append([-C(2,'right',0)])
clauses.append([-C(3,'right',0)])
# Boat starts on the left
clauses.append([B(0)])

# Goal conditions (all entities must be on the right bank at the last time step)
clauses.append([M(1,'right',T-1)])
clauses.append([M(2,'right',T-1)])
clauses.append([M(3,'right',T-1)])
clauses.append([C(1,'right',T-1)])
clauses.append([C(2,'right',T-1)])
clauses.append([C(3,'right',T-1)])

# Movement constraints for missionaries and cannibals
for t in range(T-1):
    for i in idx:
        # Define constraints to ensure valid transitions between positions
            # - Ensure individuals do not teleport between sides
            # - Ensure consistency in movements
            # - Ensure valid boat boarding logic

        # M(i, 'boat', t) and B(t+1) => -M(i, 'right', t+1)
        # M(i, 'boat', t) and -B(t+1) => -M(i, 'left', t+1)
        clauses.append([-M(i, 'boat', t),-B(t+1),-M(i, 'right', t+1)])
        clauses.append([-C(i, 'boat', t),-B(t+1),-C(i, 'right', t+1)])

        clauses.append([-M(i, 'boat', t),B(t+1),-M(i, 'left', t+1)])
        clauses.append([-C(i, 'boat', t),B(t+1),-C(i, 'left', t+1)])

        #  M(i, 'boat', t) and B(t) => M(i, 'right', t+1) or M(i, 'boat', t+1)
        #  M(i, 'boat', t) and -B(t) => M(i, 'left', t+1) or M(i, 'boat', t+1)
        clauses.append([-M(i, 'boat', t),-B(t),M(i, 'right', t+1),M(i, 'boat', t+1)])
        clauses.append([-C(i, 'boat', t),-B(t),M(i, 'right', t+1),C(i, 'boat', t+1)])

        clauses.append([-M(i, 'boat', t),B(t),M(i, 'left', t+1),M(i, 'boat', t+1)])
        clauses.append([-C(i, 'boat', t),B(t),M(i, 'left', t+1),C(i, 'boat', t+1)])

        # M(i, 'left', t) and  -B(t)=> -M(i, 'right', t + 1)
        clauses.append([-M(i, 'left', t), B(t), -M(i, 'right', t + 1)])
        clauses.append([-C(i, 'left', t), B(t), -C(i, 'right', t + 1)])

        # M(i, 'right', t) and  B(t)=> -M(i, 'left', t + 1)
        clauses.append([-M(i, 'right', t), -B(t), -M(i, 'left', t + 1)])
        clauses.append([-C(i, 'right', t), -B(t), -C(i, 'left', t + 1)])


        # M(i, 'left', t) and -B(t + 1) => M(i, 'left', t + 1)
        # M(i, 'right', t) and B(t + 1) => M(i, 'right', t + 1)
        clauses.append([-M(i, 'left', t),B(t + 1),M(i, 'left', t + 1)])
        clauses.append([-C(i, 'left', t),B(t + 1),C(i, 'left', t + 1)])
        clauses.append([-M(i, 'right', t),-B(t + 1),M(i, 'right', t + 1)])
        clauses.append([-C(i, 'right', t),-B(t + 1),C(i, 'right', t + 1)])

        #  M(i,'right',t+1) and  -M(i,'right',t) => M(i, 'boat', t)
        clauses.append([-M(i,'right',t+1),M(i,'right',t),M(i, 'boat', t)])
        clauses.append([-C(i,'right',t+1),C(i,'right',t),C(i, 'boat', t)])
        #  M(i,'left',t+1) and  -M(i,'left',t) => M(i, 'boat', t)
        clauses.append([-M(i,'left',t+1),M(i,'left',t),M(i, 'boat', t)])
        clauses.append([-C(i,'left',t+1),C(i,'left',t),C(i, 'boat', t)])

# Constraints to prevent cannibals from outnumbering missionaries on any bank

# NOTE In the original version of the problem, the rule states that on either bank, if
# any missionaries are present, they must not be outnumbered by cannibals. However,
# in the version I encode, the constraint is: on either bank, the number of
# missionaries must not be less than the number of cannibals—regardless of whether
# missionaries are present or not. For example, in the original version, having 0 
# missionaries and 1 cannibal on a bank is allowed, but under this encoding, it is not.
# Still, any solution that satisfies this version is also a valid solution to
# the original problem.

# Left bank
for t in range(T):
    Ms_left=[M(i,'left',t) for i in idx]
    Cs_left=[C(i,'left',t) for i in idx]
    cnf1 = PBEnc.geq(
        top_id=top_id,
        lits=Ms_left+Cs_left,
        weights=[1 for _ in range(len(idx))] + [-1 for _ in range(len(idx))],
        bound=0
    )
    clauses.extend(cnf1.clauses)
    top_id = cnf1.nv
# Right bank
    Ms_right=[M(i,'right',t) for i in idx]
    Cs_right=[C(i,'right',t) for i in idx]
    cnf1 = PBEnc.geq(
        top_id=top_id,
        lits=Ms_right+Cs_right,
        weights=[1 for _ in range(len(idx))] + [-1 for _ in range(len(idx))],
        bound=0
    )
    clauses.extend(cnf1.clauses)
    top_id = cnf1.nv


# Each individual must be in exactly one position at any time step
for t in range(T):
    for i in idx:
        clauses.append([M(i,p,t) for p in positions])
        clauses.append([C(i,p,t) for p in positions])
        for p1 in positions:
            for p2 in positions:
                if p1!=p2:
                    clauses.append([-M(i,p1,t),-M(i,p2,t)])
                    clauses.append([-C(i, p1, t), -C(i, p2, t)])


# Boat has a capacity of at most two people
for t in range(T):
    Ms=[M(i,'boat',t) for i in idx]
    Cs=[C(i,'boat',t) for i in idx]
    cnf = CardEnc.atmost(lits=Ms+Cs,bound=2, encoding=EncType.seqcounter,top_id=top_id)
    top_id=cnf.nv
    for clause in cnf.clauses:
        clauses.append(clause)

# Boat cannot travel empty
for t in range(T-1):
    clauses.append([M(i,'boat',t) for i in idx]+[C(i,'boat',t) for i in idx])

res=None
cnf = CNF(from_clauses=clauses)
with Solver(name='cadical153', bootstrap_with=cnf) as solver:
    if solver.solve():
        print("SAT")
        res = [[0,0,0,0,0,0] for _ in range(T)]
        model = solver.get_model()
        # Parse SAT solver results
        for m in model:
            if m > 0 and m in vpool.id2obj:
                parts = vpool.id2obj[m].split('@')
                if parts[0]=="Missionary":
                    if parts[2]=="left":
                        res[int(parts[3])][0]+=1
                    elif parts[2]=="boat":
                        res[int(parts[3])][2]+=1
                    elif parts[2]=="right":
                        res[int(parts[3])][4]+=1
                elif parts[0]=="Cannibal":
                    if parts[2]=="left":
                        res[int(parts[3])][1]+=1
                    elif parts[2]=="boat":
                        res[int(parts[3])][3]+=1
                    elif parts[2]=="right":
                        res[int(parts[3])][5]+=1
    else:
        print("UNSAT")

if res is not None:
    new_res=[]
    boat_location=[1,1]
    init_state = (3, 3, 0, 0, 0, 0)
    print(
        f'({init_state[0]},{init_state[1]})' + ' ' * 5 + f'🚣({init_state[2]},{init_state[3]})' + ' ' * 5 + f'({init_state[4]},{init_state[5]})')
    print(
        f'({res[0][0]},{res[0][1]})' + ' ' * 5 + f'🚣({res[0][2]},{res[0][3]})' + ' ' * 5 + f'({res[0][4]},{res[0][5]})')

    new_res.append([(init_state[0],init_state[1]),(init_state[2],init_state[3]),(0,0),(init_state[4],init_state[5])])
    new_res.append([(res[0][0],res[0][1]),(res[0][2],res[0][3]),(0,0),(res[0][4],res[0][5])])


    for t in range(1, T):
        if t % 2 == 0:
            print(
                f'({res[t - 1][0]},{res[t - 1][1]})' + ' ' * 5 + f'🚣({res[t - 1][2]},{res[t - 1][3]})' + ' ' * 5 + f'({res[t][4]},{res[t][5]})')
            print(
                f'({res[t][0]},{res[t][1]})' + ' ' * 5 + f'🚣({res[t][2]},{res[t][3]})' + ' ' * 5 + f'({res[t][4]},{res[t][5]})')

            new_res.append([(res[t - 1][0], res[t - 1][1]), (res[t - 1][2], res[t - 1][3]), (0, 0), (res[t][4], res[t][5])])
            new_res.append([(res[t][0], res[t][1]), (res[t][2], res[t][3]), (0, 0), (res[t][4], res[t][5])])
            boat_location.append(1)
            boat_location.append(1)
        else:
            print(
                f'({res[t][0]},{res[t][1]})' + ' ' * 5 + f'  ({res[t - 1][2]},{res[t - 1][3]})🚣' + ' ' * 5 + f'({res[t - 1][4]},{res[t - 1][5]})')
            print(
                f'({res[t][0]},{res[t][1]})' + ' ' * 5 + f'  ({res[t][2]},{res[t][3]})🚣' + ' ' * 5 + f'({res[t][4]},{res[t][5]})')
            new_res.append([(res[t][0], res[t][1]),(0, 0), (res[t - 1][2], res[t - 1][3]), (res[t-1][4], res[t-1][5])])
            new_res.append([(res[t][0], res[t][1]),(0, 0), (res[t][2], res[t][3]), (res[t][4], res[t][5])])
            boat_location.append(2)
            boat_location.append(2)

    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # Windows (Assuming PLB works)
    # Uncomment the lines below if using Windows with proper font support
    # plt.rcParams['font.family'] = 'Segoe UI Emoji'
    # missionary_emoji = "🧑‍"
    # cannibal_emoji = "👹"

    # Linux
    matplotlib.use('TkAgg')
    missionary_emoji = "M"
    cannibal_emoji = "C"

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)

    state_text = ax.text(1.5, 1.5, "", fontsize=65, ha='center', va='center', fontweight='bold')


    # Function to update the state at each time step
    def update_frame(t):
        state = new_res[t]
        s1 = missionary_emoji * state[0][0] + cannibal_emoji * state[0][1]
        s2 = missionary_emoji * state[1][0] + cannibal_emoji * state[1][1]
        s3 = missionary_emoji * state[2][0] + cannibal_emoji * state[2][1]
        s4 = missionary_emoji * state[3][0] + cannibal_emoji * state[3][1]

        if boat_location[t] == 1:
            s2 = "|" + s2 + "|"
        else:
            s3 = "|" + s3 + "|"

        state_text.set_text(s1 + "\n" + s2 + "\n" + s3 + "\n" + s4)


    ani = animation.FuncAnimation(fig, update_frame, frames=len(new_res), interval=1000, repeat=False)

    plt.show()
