import pysat.card
from pysat.card import CardEnc
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver
from pysat.pb import *
import networkx as nx
from networkx.algorithms.approximation import greedy_tsp

"""
This script demonstrates a potential use case of SAT solver interfaces, employing dynamic clause addition lazily during
the SAT solving process, applied to solving the Traveling Salesman Problem (TSP).

Steps:
1. We express the key properties of the TSP (e.g., each city must appear exactly once in the tour) 
   in logical form.
2. We pass this representation to the SAT solver to find a Hamiltonian cycle.
3. We compute the cost of the resulting cycle and store it as COST.
4. We add a new constraint specifying that we are only interested in tours with a cost less than COST.

The main idea is that, when adding a constraint in step 4, we do not create a new solver object. 
Instead, we modify the existing solver from the previous call. This allows the solver to reuse parts 
of its state from earlier calls, such as learned clauses and internal search history, which can improve efficiency.

This approach is made possible by the SAT solver's API, enabling users to interact with it at a high level 
without needing to understand the internal workings of the solver.

It is also worth noting, however, that there are other interfaces, such as the PASIR-UP interface, that can be useful 
in similar or different settings, such as satisfiability modulo theories (SMT) and lazy clause generation (LCG). 
However, these interfaces may require more in-depth knowledge of the solver's internals compared to a black-box 
approach.

Interested readers can refer to the paper proposing interfaces like PASIR-UP and consult the pysat.engines section in 
the PySAT documentation for further information.
"""

# NOTE: Assumes that all cost values are integers and self-loops are prohibited.

# --- Problem Setup ---

# Distance matrix from Google OR-Tools TSP example (13 cities)
cost = [
    [0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972],
    [2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579],
    [713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260],
    [1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987],
    [1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586, 371],
    [1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887, 999],
    [2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1724, 1891, 1114, 701],
    [213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1605, 2300, 2099],
    [2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653, 600],
    [875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272, 1162],
    [1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, 0, 1017, 1200],
    [2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, 0, 504],
    [1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, 0],
]


N=len(cost)

# --- Variable Pool and Helper Functions ---
vpool = IDPool(start_from=1)
V = lambda i, t: vpool.id(f'V@{i}@{t}')  # city i is at position t
E = lambda i,j: vpool.id(f'E@{i}@{j}')  # edge (i → j) is used

cnf=CNF()

# --- SAT Encoding ---

# Constraint 1: Each city must appear exactly once in the tour
for c in range(N):
    cnf.extend(CardEnc.equals(lits=[V(c, t) for t in range(N)], bound=1, vpool=vpool, encoding=pysat.card.EncType.pairwise))

# Constraint 2: Each position must be occupied by exactly one city
for t in range(N):
    cnf.extend(CardEnc.equals(lits=[V(c, t) for c in range(N)], bound=1, vpool=vpool, encoding=pysat.card.EncType.pairwise))


# Constraint 3: If city i is at position t and city j is at position t+1, then edge (i → j) is used
for t in range(N):
    for i in range(N):
        for j in range(N):
            if i != j:
                cnf.append([-V(i,t),-V(j,(t+1)%N),E(i,j)])

# Constraint 4: Exactly N edges should be used (a Hamiltonian cycle has N edges)
cnf.extend(CardEnc.equals(lits=[E(i,j) for i in range(N) for j in range(N) if i!=j], bound=N, vpool=vpool, encoding=pysat.card.EncType.totalizer))

# --- Upper Bound Estimation using Greedy TSP ---

# Build graph with edge weights from cost matrix
G = nx.complete_graph(N)
for i in range(N):
    for j in range(i+1, N):
        G[i][j]['weight'] = cost[i][j]
        G[j][i]['weight'] = cost[j][i]

# Get an initial greedy tour to compute a cost bound
cycle = greedy_tsp(G, weight='weight', source=0)
bound = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle))

# Add pseudo-boolean constraint for cost upper bound
cnf.extend(PBEnc.leq(
    lits=[E(i, j) for i in range(N) for j in range(N) if i != j],
    weights=[cost[i][j] for i in range(N) for j in range(N) if i != j],
    bound=bound - 1,
    vpool=vpool
))

print(bound, cycle)

# --- SAT Solving Loop for Improving Solutions ---
with Solver(name='cadical153', bootstrap_with=cnf) as solver:
    while solver.solve():
        # Decode tour from model
        Tour=[None]*N
        cnt_None=N
        model = solver.get_model()
        for m in model:
            if m>0 and m in vpool.id2obj:
                item=vpool.id2obj[m].split('@')
                if item[0]=="V":
                    city, pos = int(item[1]), int(item[2])
                    Tour[pos] = city
                    cnt_None= cnt_None - 1
                if cnt_None==0:
                    break
        # Calculate tour cost
        bound = sum(cost[Tour[i]][Tour[(i + 1) % N]] for i in range(N))
        print(bound,Tour+[Tour[0]])
        # Add constraint to find strictly better tour in next iterations
        solver.append_formula(PBEnc.leq(
            lits=[E(i, j) for i in range(N) for j in range(N) if i != j],
            weights=[cost[i][j] for i in range(N) for j in range(N) if i != j],
            bound=bound - 1,
            vpool=vpool
        ))




"""
Without NetworkX (no initial bound pruning):
18703 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
17612 [0, 4, 6, 10, 9, 5, 12, 8, 7, 2, 1, 3, 11]
.
.
.
7293 [3, 2, 7, 0, 9, 5, 10, 11, 1, 8, 6, 12, 4]
"""

"""
With NetworkX Greedy TSP bound:
8131 [0, 7, 2, 9, 3, 4, 12, 11, 1, 8, 6, 5, 10, 0]
8107 [7, 2, 3, 9, 10, 5, 4, 12, 6, 8, 1, 11, 0, 7]
.
.
.
7293 [9, 0, 7, 2, 3, 4, 12, 6, 8, 1, 11, 10, 5, 9]
"""

"""
Summary:

Without NetworkX (no initial bound pruning):
  Initial cost: 18703
  Final optimal cost: 7293
  Steps to optimal: 29
  Final tour: [3, 2, 7, 0, 9, 5, 10, 11, 1, 8, 6, 12, 4]

With NetworkX Greedy TSP bound:
  Initial cost: 8131
  Final optimal cost: 7293
  Steps to optimal: 8
  Final tour: [9, 0, 7, 2, 3, 4, 12, 6, 8, 1, 11, 10, 5, 9]
"""