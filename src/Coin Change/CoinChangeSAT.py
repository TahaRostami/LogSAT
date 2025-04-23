from pysat.formula import CNF, IDPool
from pysat.solvers import Solver
from pysat.card import CardEnc
from pysat.pb import PBEnc, EncType
from rich import print
from rich.text import Text

"""
Problem Definition:
-------------------
Given:
    - A list of coin denominations: coins = [1, 2, 5]
    - A target amount of money: A = 4

Objective:
    - Show the different ways to select any number of coins (with an unlimited supply of each denomination) such that 
       their total sum equals exactly A.

This script uses PySAT to encode and solve the problem via model enumeration.
"""

A = 9  # total amount of money
coins = [2, 3, 5, 6]  # available coin denominations

vpool = IDPool(start_from=1)

# C(d, k) = True if exactly k coins of denomination d are chosen
C = lambda d, k: vpool.id(f"C@{d}@{k}")

# Determine upper bound (max number of coins) for each denomination
upper_bounds = [A // c for c in coins]

cnf = CNF()

# Ensure exactly one quantity is chosen for each denomination: 0, 1, ..., upper_bound[i]
for i, coin in enumerate(coins):
    choices = [C(coin, j) for j in range(upper_bounds[i] + 1)]
    cnf.extend(CardEnc.equals(lits=choices, bound=1, encoding=EncType.best, vpool=vpool).clauses)

# Encode total sum constraint: sum of k * d over all choices == A
lits = []
weights = []
for i, coin in enumerate(coins):
    for k in range(upper_bounds[i] + 1):
        lits.append(C(coin, k))
        weights.append(k * coin)

cnf.extend(PBEnc.equals(lits=lits, weights=weights, bound=A, encoding=EncType.sortnetwrk, vpool=vpool).clauses)

# Print the problem setup using rich
print(f"\n\t\t\t   [bold underline]Solutions to the following equation:[/bold underline]\n")

# Function to convert a number to its subscript version
def subscript_number(num):
    subscript_map = {
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅',
        '6': '₆', '7': '₇', '8': '₈', '9': '₉'
    }
    return ''.join(subscript_map.get(digit, digit) for digit in str(num))

equation_text = " + ".join([f"k{subscript_number(i)}x{coin}" for i, coin in enumerate(coins)]) + f" = {A}"
print(Text("\t\t\t\t   "+equation_text))

for j in range(len(upper_bounds)):
    print("\t\t\t\t\t"+f"[bold grey]0 <= k{subscript_number(j)} <= {upper_bounds[j]}[/bold grey]")

# Model enumeration with SAT solver
print(f"\n\t\t\t   [bold underline]Combinations that sum up to {A}:[/bold underline]\n")

# Store all solutions for potential further processing or analysis
solutions = []
with Solver(name='cadical153', bootstrap_with=cnf) as solver:
    while solver.solve():
        model = solver.get_model()
        total = 0
        parts = []
        used_coins = []

        for m in model:
            if m > 0 and m in vpool.id2obj:
                _, d, k = vpool.id2obj[m].split("@")
                d, k = int(d), int(k)
                if k > 0:
                    used_coins.append((k, d))
                    total += k * d

        line = Text()
        line.append("\t\t\t\t\t")
        for idx, (k, d) in enumerate(used_coins):
            part = Text(f"{k}x")
            part.append(str(d), style="bold green")
            line.append(part)
            if idx != len(used_coins) - 1:
                line.append(" + ")
        line.append(f" = {total}")

        print(line)
        solutions.append(used_coins)

        # When the solver returns a model, it includes assignments to all variables—both original variables
        # (e.g., C(d, k)) and auxiliary variables introduced during encoding (e.g., for cardinality and Pseudo-Boolean
        # constraints). Here, what we really want is to prevent the solver from returning the same assignment
        # to the relevant original variables.
        # Therefore, instead of blocking the entire model—which results in a long and overly specific clause—
        # we can block only the assignment to the relevant original variables, yielding a shorter and more
        # general clause. The following line demonstrates this:
        solver.add_clause([-C(d, k) for k, d in used_coins])


"""
# NOTE: PySAT also provides `solver.enum_models()` for model enumeration, which sometimes can simplify the process.
# Here's an alternative version using that approach:

solutions = set()

with Solver(name='cadical153', bootstrap_with=cnf) as solver:
    for model in solver.enum_models():
        used_coins = []
        total = 0
        for m in model:
            if m > 0 and m in vpool.id2obj:
                _, d, k = vpool.id2obj[m].split("@")
                d, k = int(d), int(k)
                if k > 0:
                    used_coins.append((k, d))
                    total += k * d
        if tuple(sorted(used_coins)) not in solutions:
            line = Text()
            line.append("\t\t\t\t\t")
            for idx, (k, d) in enumerate(used_coins):
                part = Text(f"{k}x")
                part.append(str(d), style="bold green")
                line.append(part)
                if idx != len(used_coins) - 1:
                    line.append(" + ")
            line.append(f" = {total}")
            print(line)
            solutions.add(tuple(sorted(used_coins)))


"""

"""
(OPTIONAL)

I did not go into probability theory here, but an interested reader might pause to consider how SAT solvers can be used
for computing probabilities in certain contexts—particularly through model enumeration.

For example, we could transform our original question into a probabilistic one:  
"What is the probability that a solution includes at least one coin of denomination 2?"

By enumerating all valid models and counting how many of them satisfy this property, SAT solvers provide a
direct method to calculate such probabilities.

This illustrates how logical reasoning tools like SAT can offer insights not only into feasibility but also into 
likelihood within well-defined constraints.
"""
