from pysat.formula import IDPool
from pysat.pb import *
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

# Input: Profit and Weight of each item, and maximum allowable weight (capacity W)
# Assumption: Each profit value in the list is positive (>0)
Profit = [300, 200, 400, 500]   # Profit values for each item
Weight = [2, 1, 5, 3]   # Corresponding weight values for each item
W = 10                # Maximum allowable total weight (capacity of the knapsack)

vpool = IDPool(start_from=1)
# whether item i-th is chosen
I = lambda i: vpool.id(f'I{i}')
for i in range(len(Profit)):
    I(i)

wcnf = WCNF()

for clause in PBEnc.leq(lits=[I(i) for i in range(len(Profit))], weights=Weight, bound=W).clauses:
    wcnf.append(clause)

for i in range(len(Profit)):
    wcnf.append([I(i)], weight=Profit[i])

with RC2(wcnf) as rc2:
    model = rc2.compute()
    total_W,total_P = 0,0
    if model is not None:
        print("SAT")
        for m in model:
            if m > 0 and m in vpool.id2obj:
                idx = int(vpool.id2obj[m][1:])
                print(f"Item (0-indexing): {idx} is chosen with weight: {Weight[idx]}, and profit: {Profit[idx]}")
                total_W, total_P=total_W+Weight[idx], total_P+ Profit[idx]
        print(f"\nTotal Weight: {total_W} (Capacity: {W}), Total Profit: {total_P}")

    else:
        print("UNSAT")
