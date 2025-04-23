from pysat.formula import IDPool
from pysat.pb import *
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

# Lengths of the rods and the associated prices (P)
# P = [0, 1, 5, 8, 9, 10, 17, 17, 20, 24, 30]
# L = [0, 1, 2, 3, 4,  5,  6, 7,   8,  9, 10]

P=[0]+ [1, 5, 8, 9, 10, 17, 17, 21]
L=list(range(0,len(P)))

vpool = IDPool(start_from=1)

# Variable representing whether the solution includes k rods of length l
C = lambda l, k: vpool.id(f'C{l}@{k}')

# Generate variables for each rod length l and the possible number of rods k
for l in L[1:]:
    for k in range((L[-1] // l) + 1):
        C(l, k)

# Initialize the Weighted CNF (WCNF) formula
wcnf = WCNF()

# Ensure that for each rod length l, we select exactly one value for the number of rods k
for l in L[1:]:
    wcnf.append([C(l, k) for k in range((L[-1] // l) + 1)])

    # Add pairwise exclusion clauses for selecting multiple k values for the same rod length
    for k1 in range((L[-1] // l) + 1):
        for k2 in range((L[-1] // l) + 1):
            if k1 != k2:
                wcnf.append([-C(l, k1), -C(l, k2)])

weights = []
lits = []

# Ensure that the sum of C(l,k) * (l*k) equals the total length of the rod before any cut
# This ensures that the cuts result in the correct total length of the rod
for l in L[1:]:
    for k in range((L[-1] // l) + 1):
        weights.append(l * k)  # The cost is based on length * number of rods
        lits.append(C(l, k))

# Add a clause to ensure that the sum of the weights equals the initial rod length before cutting
for clause in PBEnc.equals(lits=lits, weights=weights, bound=L[-1]).clauses:
    wcnf.append(clause)

# Maximize the total price by adding soft clauses for selecting rods with positive prices
for l in L[1:]:
    for k in range((L[-1] // l) + 1):
        # Only add a soft clause if the price (P[l] * k) is greater than 0
        # It is important to exclude weight 0, as it would treat those variables as hard clauses
        # This would incorrectly force all variables with zero quantity to be satisfied, which is wrong
        if P[l] * k > 0:
            wcnf.append([C(l, k)], weight=P[l] * k)

# Solve the MaxSAT problem using the RC2 solver
with RC2(wcnf) as rc2:
    model = rc2.compute()

    if model is not None:
        print("SAT: Solution found!")
        total_price = 0
        solution = []

        # Process the model and print the selected rod lengths and quantities
        for m in model:
            if m > 0 and m in vpool.id2obj:
                l, k = vpool.id2obj[m].split('@')
                l, k = int(l[1:]), int(k)
                solution.append((l, k))
                total_price += P[l] * k

        print("\nSelected Rods (Length, Quantity):")
        for (l, k) in solution:
            if k > 0:
                print(f"Length {l}, Quantity {k}")

        print("\nTotal Price of Solution:", total_price)

    else:
        print("UNSAT: No solution found.")
