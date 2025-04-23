from pysat.formula import IDPool
from pysat.card import *
from pysat.solvers import Solver

moves = [
    'top_right',  # Circulate the top part in the right direction horizontally (once)
    'bottom_right',  # Circulate the bottom part in the right direction horizontally (once)
    'left_up',  # Circulate the left part in the upward direction vertically (once)
    'right_up',  # Circulate the right part in the upward direction vertically (once)
    'nothing'  # No move, leave the cube unchanged
]

colors=['white','orange','green','red','blue', 'yellow']
max_T=11

vpool = IDPool(start_from=1)
# Indicates whether sticker (i,j,k) at time t has color c
S = lambda i, j, k, t, c: vpool.id(f'S@{i}@{j}@{k}@{t}@{c}')
for i in range(2):
       for j in range(2):
              for k in range(6):
                     for t in range(max_T):
                            for c in colors:
                                   S(i,j,k,t,c)

# Indicates whether move m at time t has been performed
M= lambda m,t: vpool.id(f'M@{m}@{t}')
for m in moves:
       for t in range(max_T):
              M(m,t)

clauses=[]

# In each time step t, exactly one move m must be chosen
for t in range(max_T):
       clauses.append([M(m,t) for m in moves])
       for m1 in moves:
              for m2 in moves:
                     if m1!=m2:
                            clauses.append([-M(m1,t),-M(m2,t)])

# In each time step t, each sticker must have exactly one color
for t in range(max_T):
       for i in range(2):
              for j in range(2):
                     for k in range(6):
                            clauses.append([S(i,j,k,t,c) for c in colors])

                            for c1 in colors:
                                   for c2 in colors:
                                          if c1!=c2:
                                                 clauses.append([-S(i,j,k,t,c1),-S(i,j,k,t,c2)])

# At each time step t, there must be exactly 4 stickers of each color
top_id=max([max(clause) for clause in clauses])
for t in range(max_T):
       for c in colors:
              all_stickers=[S(i,j,k,t,c) for i in range(2) for j in range(2) for k in range(6)]
              cnf = CardEnc.equals(lits=all_stickers, encoding=EncType.seqcounter, bound=4,top_id=top_id)
              for clause in cnf.clauses:
                     clauses.append(clause)
              top_id=cnf.nv

# Goal state: For each face in the last time step, all stickers on that face must have the same color
for k in range(6):
    for c in colors:
        for i in range(2):
            for j in range(2):
                if (i,j)!=(0,0):
                    clauses.append([-S(0,0,k,max_T-1,c),S(i,j,k,max_T-1,c)])




for t in range(max_T-1):
    for c in colors:

            # region 'top_right'
            clauses.append([-M('top_right', t), -S(0, 0, 0, t, c), S(1, 0, 0, t+1, c)])
            clauses.append([-M('top_right', t), -S(0, 1, 0, t, c), S(0, 0, 0, t + 1, c)])
            clauses.append([-M('top_right', t), -S(1, 0, 0, t, c), S(1, 1, 0, t + 1, c)])
            clauses.append([-M('top_right', t), -S(1, 1, 0, t, c), S(0, 1, 0, t + 1, c)])

            for j in range(2):
                for k in [1,2,3,4]:
                    clauses.append([-M('top_right',t),-S(0,j,k,t,c),S(0,j,(k%4)+ 1,t+1,c)])

            for j in range(2):
                for k in [1,2,3,4]:
                    clauses.append([-M('top_right',t),-S(1,j,k,t,c),S(1,j,k,t+1,c)])
            for i in range(2):
                for j in range(2):
                         clauses.append([-M('top_right',t),-S(1,j,5,t,c),S(1,j,5,t+1,c)])


            # region 'bottom_right'

            for i in range(2):
                for j in range(2):
                    clauses.append([-M('bottom_right', t), -S(1, j, 0, t, c), S(1, j, 0, t + 1, c)])

            for j in range(2):
                for k in [1, 2, 3, 4]:
                    clauses.append([-M('bottom_right', t), -S(0, j, k, t, c), S(0, j, k, t + 1, c)])


            for j in range(2):
                for k in [1, 2, 3, 4]:
                    clauses.append([-M('bottom_right', t), -S(1, j, k, t, c), S(1, j, (k % 4) + 1, t + 1, c)])

            clauses.append([-M('bottom_right', t), -S(0, 0, 5, t, c), S(0, 1, 5, t + 1, c)])
            clauses.append([-M('bottom_right', t), -S(0, 1, 5, t, c), S(1, 1, 5, t + 1, c)])
            clauses.append([-M('bottom_right', t), -S(1, 0, 5, t, c), S(0, 0, 5, t + 1, c)])
            clauses.append([-M('bottom_right', t), -S(1, 1, 5, t, c), S(1, 0, 5, t + 1, c)])

            # region 'left_up'
            clauses.append([-M('left_up', t), -S(0, 0, 2, t, c), S(0, 0, 0, t + 1, c)])
            clauses.append([-M('left_up', t), -S(1, 0, 2, t, c), S(1, 0, 0, t + 1, c)])
            clauses.append([-M('left_up', t), -S(0, 0, 5, t, c), S(0, 0, 2, t + 1, c)])
            clauses.append([-M('left_up', t), -S(1, 0, 5, t, c), S(1, 0, 2, t + 1, c)])
            clauses.append([-M('left_up', t), -S(0, 0, 0, t, c), S(1, 1, 4, t + 1, c)])
            clauses.append([-M('left_up', t), -S(1, 0, 5, t, c), S(0, 1, 4, t + 1, c)])
            clauses.append([-M('left_up', t), -S(0, 1, 4, t, c), S(1, 0, 5, t + 1, c)])
            clauses.append([-M('left_up', t), -S(1, 1, 4, t, c), S(0, 0, 5, t + 1, c)])

            clauses.append([-M('left_up', t), -S(0, 0, 1, t, c), S(1, 0, 1, t + 1, c)])
            clauses.append([-M('left_up', t), -S(0, 1, 1, t, c), S(0, 0, 1, t + 1, c)])
            clauses.append([-M('left_up', t), -S(1, 0, 1, t, c), S(1, 1, 1, t + 1, c)])
            clauses.append([-M('left_up', t), -S(1, 1, 1, t, c), S(0, 1, 1, t + 1, c)])

            clauses.append([-M('left_up', t), -S(0, 1, 0, t, c), S(0, 1, 0, t + 1, c)])
            clauses.append([-M('left_up', t), -S(1, 1, 0, t, c), S(1, 1, 0, t + 1, c)])
            clauses.append([-M('left_up', t), -S(0, 1, 2, t, c), S(0, 1, 2, t + 1, c)])
            clauses.append([-M('left_up', t), -S(1, 1, 2, t, c), S(1, 1, 2, t + 1, c)])
            clauses.append([-M('left_up', t), -S(0, 1, 5, t, c), S(0, 1, 5, t + 1, c)])
            clauses.append([-M('left_up', t), -S(1, 1, 5, t, c), S(1, 1, 5, t + 1, c)])
            clauses.append([-M('left_up', t), -S(0, 0, 4, t, c), S(0, 0, 4, t + 1, c)])
            clauses.append([-M('left_up', t), -S(1, 0, 4, t, c), S(1, 0, 4, t + 1, c)])
            clauses.append([-M('left_up', t), -S(0, 0, 3, t, c), S(0, 0, 3, t + 1, c)])
            clauses.append([-M('left_up', t), -S(0, 1, 3, t, c), S(0, 1, 3, t + 1, c)])
            clauses.append([-M('left_up', t), -S(1, 0, 3, t, c), S(1, 0, 3, t + 1, c)])
            clauses.append([-M('left_up', t), -S(1, 1, 3, t, c), S(1, 1, 3, t + 1, c)])

            # region 'right_up'
            clauses.append([-M('right_up', t), -S(0, 1, 2, t, c), S(0, 1, 0, t + 1, c)])
            clauses.append([-M('right_up', t), -S(1, 1, 2, t, c), S(1, 1, 0, t + 1, c)])
            clauses.append([-M('right_up', t), -S(0, 1, 5, t, c), S(0, 1, 2, t + 1, c)])
            clauses.append([-M('right_up', t), -S(1, 1, 5, t, c), S(1, 1, 2, t + 1, c)])
            clauses.append([-M('right_up', t), -S(0, 1, 0, t, c), S(1, 0, 4, t + 1, c)])
            clauses.append([-M('right_up', t), -S(1, 1, 0, t, c), S(0, 0, 4, t + 1, c)])
            clauses.append([-M('right_up', t), -S(0, 0, 4, t, c), S(1, 1, 5, t + 1, c)])
            clauses.append([-M('right_up', t), -S(1, 0, 4, t, c), S(0, 1, 5, t + 1, c)])

            clauses.append([-M('right_up', t), -S(0, 0, 3, t, c), S(0, 1, 3, t + 1, c)])
            clauses.append([-M('right_up', t), -S(0, 1, 3, t, c), S(1, 1, 3, t + 1, c)])
            clauses.append([-M('right_up', t), -S(1, 0, 3, t, c), S(0, 0, 3, t + 1, c)])
            clauses.append([-M('right_up', t), -S(1, 1, 3, t, c), S(1, 0, 3, t + 1, c)])

            clauses.append([-M('right_up', t), -S(0, 0, 0, t, c), S(0, 0, 0, t + 1, c)])
            clauses.append([-M('right_up', t), -S(1, 0, 0, t, c), S(1, 0, 0, t + 1, c)])
            clauses.append([-M('right_up', t), -S(0, 0, 2, t, c), S(0, 0, 2, t + 1, c)])
            clauses.append([-M('right_up', t), -S(1, 0, 2, t, c), S(1, 0, 2, t + 1, c)])
            clauses.append([-M('right_up', t), -S(0, 0, 5, t, c), S(0, 0, 5, t + 1, c)])
            clauses.append([-M('right_up', t), -S(1, 0, 5, t, c), S(1, 0, 5, t + 1, c)])
            clauses.append([-M('right_up', t), -S(0, 1, 4, t, c), S(0, 1, 4, t + 1, c)])
            clauses.append([-M('right_up', t), -S(1, 1, 4, t, c), S(1, 1, 4, t + 1, c)])
            clauses.append([-M('right_up', t), -S(0, 0, 1, t, c), S(0, 0, 1, t + 1, c)])
            clauses.append([-M('right_up', t), -S(0, 1, 1, t, c), S(0, 1, 1, t + 1, c)])
            clauses.append([-M('right_up', t), -S(1, 0, 1, t, c), S(1, 0, 1, t + 1, c)])
            clauses.append([-M('right_up', t), -S(1, 1, 1, t, c), S(1, 1, 1, t + 1, c)])


            # region 'nothing'
            for i in range(2):
                for j in range(2):
                    for k in range(6):
                        clauses.append([-M('nothing', t), -S(i, j, k, t, c), S(i, j, k, t + 1, c)])


#initialization

#face 2
clauses.append([S(0, 0, 2, 0, 'white')])
clauses.append([S(0, 1, 2, 0, 'yellow')])
clauses.append([S(1, 0, 2, 0, 'blue')])
clauses.append([S(1, 1, 2, 0, 'orange')])

#face 0
clauses.append([S(0, 0, 0, 0, 'orange')])
clauses.append([S(0, 1, 0, 0, 'green')])
clauses.append([S(1, 0, 0, 0, 'orange')])
clauses.append([S(1, 1, 0, 0, 'red')])

#face 1
clauses.append([S(0, 0, 1, 0, 'green')])
clauses.append([S(0, 1, 1, 0, 'green')])
clauses.append([S(1, 0, 1, 0, 'green')])
clauses.append([S(1, 1, 1, 0, 'white')])

#face 3
clauses.append([S(0, 0, 3, 0, 'blue')])
clauses.append([S(0, 1, 3, 0, 'red')])
clauses.append([S(1, 0, 3, 0, 'blue')])
clauses.append([S(1, 1, 3, 0, 'yellow')])

#face 4
clauses.append([S(0, 0, 4, 0, 'white')])
clauses.append([S(0, 1, 4, 0, 'yellow')])
clauses.append([S(1, 0, 4, 0, 'blue')])
clauses.append([S(1, 1, 4, 0, 'yellow')])

#face 5
clauses.append([S(0, 0, 5, 0, 'red')])
clauses.append([S(0, 1, 5, 0, 'white')])
clauses.append([S(1, 0, 5, 0, 'red')])
clauses.append([S(1, 1, 5, 0, 'orange')])

print('Encoded!')


def print_cube_at_time_step(s, t):
    print("                  " + "   ".join(s[t][0][0]))
    print("                  " + "   ".join(s[t][0][1]))
    print("                   -----------")

    for row in range(2):
        print("   ".join(s[t][1][row]) + "  |  " + "   ".join(s[t][2][row]) + "  |  " + "   ".join(s[t][3][row]) + "  |  " + "   ".join(s[t][4][row]))
    print("                   -----------")

    print("                  " + "   ".join(s[t][5][0]))
    print("                  " + "   ".join(s[t][5][1]))


cnf = CNF(from_clauses=clauses)
with Solver(name='Cadical195', bootstrap_with=cnf) as solver:
    if solver.solve():
        print("SAT")
        model=solver.get_model()
        instructions = []
        s=[[[['x' for j in range(2)] for i in range(2)] for f in range(6)] for _ in range(max_T)]

        for m in model:
            if m > 0 and m in vpool.id2obj:
                item=vpool.id2obj[m].split('@')
                if item[0] == "M" and item[1] != 'nothing':
                    instructions.append(item)
                if item[0]=="S":
                    i,j,k,t,c=int(item[1]),int(item[2]),int(item[3]),int(item[4]),item[5]
                    s[t][k][i][j]=c


        instructions.sort(key=lambda x: int(x[2]))

        print_cube_at_time_step(s, 0)
        for t in range(1,max_T):
            print('\n',t,'- ', instructions[t-1][1])
            print_cube_at_time_step(s, t)

        print('-'*25)
        for t in range(1,max_T):
            print(t, '- ', instructions[t - 1][1])
    else:
        print("UNSAT")




