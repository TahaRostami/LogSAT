import subprocess
import tempfile
import os
from itertools import combinations
import pysat.formula
from pysat.formula import IDPool
from pysat.solvers import Solver

"""
This script demonstrates the potential benefits of preprocessing techniques, such as (static) symmetry breaking, in SAT
solving.
The code focuses on the Ramsey problem, using a simplified version of the encoding described in [1]. In this simplified
encoding, a SAT result indicates that R(s,t) > n, while UNSAT suggests R(s,t) ≤ n. For this specific experiment, we 
assume that R(3,7) > 22, and we are investigating whether R(3,7) = 23.
Symmetry breaking is applied using Satsuma [2], a recently introduced tool for static symmetry breaking preprocessing
for SAT solvers.
"""


vpool = IDPool(start_from=1)
n=23
s=3
t=7
print(f"Investigation of R({s},{t}) > {n}:")
vertices=[i for i in range(n)]

# Edge from i to j or vice versa (assuming i < j)
E = lambda i,j: vpool.id(f'E@{i}@{j}')

clauses=[]

# No need to enforce i < j explicitly, as the Python implementation of combinations returns results in expected order
for Ks_vertices in combinations(vertices, s):
    clauses.append([-E(e[0],e[1]) for e in combinations(Ks_vertices, 2)])

for Kt_vertices in combinations(vertices, t):
    clauses.append([E(e[0],e[1]) for e in combinations(Kt_vertices, 2)])

# At this point, the encoding is complete.
# Assuming that the Satsuma executable is located in the same directory as this script,
# the following code calls Satsuma as a command-line tool, passes the encoded formula to it, and executes the command.
# Afterward, it reads the resulting formula output from Satsuma and passes it to the SAT solver for solving.

with tempfile.NamedTemporaryFile(mode='w+', suffix='.cnf', delete=False) as tmp_input, \
     tempfile.NamedTemporaryFile(mode='r', suffix='.break.cnf', delete=False) as tmp_output:

    input_path = tmp_input.name
    output_path = tmp_output.name

    try:
        # Write the CNF formula to the input file
        cnf = pysat.formula.CNF(from_clauses=clauses)
        cnf.to_file(input_path)

        # Run Satsuma, capture stdout and stderr (stderr contains time information on success)
        result = subprocess.run(
            ["./satsuma", "-f", input_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        # Parse the time taken by Satsuma (in ms) and convert it to seconds
        satsuma_exe_time = float(result.stderr.strip().splitlines()[-2].split()[1].replace('ms', ''))/1000

        # Load the modified CNF formula from Satsuma's output
        cnf = pysat.formula.CNF(from_string=result.stdout)

        # Use the SAT solver to solve the instance
        with Solver(name='Cadical195', bootstrap_with=cnf, use_timer=True) as solver:
            if solver.solve():
                print("SAT")
            else:
                print("UNSAT")
            print(f"Symmetry breaking using Satsuma took: {satsuma_exe_time} seconds")
            print(f"Solving the instance using SAT solver took: {solver.time_accum()} seconds")

    finally:
        for file in [input_path, output_path]:
            if os.path.exists(file):
                os.remove(file)


"""
_______________________________________________________________________________________
                                        Satsuma 1.2    +  CaDiCaL 1.9.5
  Ramsey(3,7) > 23   | is UNSAT |        2 seconds     +   206 seconds    = 208 seconds    
_______________________________________________________________________________________
                                               Only CaDiCaL 1.9.5
  Ramsey(3,7) > 23   | is UNSAT |                 >1200 seconds      
_______________________________________________________________________________________                        
"""

"""
Interested readers may explore common preprocessing techniques such as blocked clause elimination, bounded variable 
elimination, and others.
For additional details, refer to the pysat.process section of the PySAT documentation, or see the Symmetry
and Satisfiability and Preprocessing in SAT Solving chapters in [3].
"""

"""
References:
[1] Z. Li, C. Duggan, C. Bright, and V. Ganesh, “Verified Certificates via SAT and Computer Algebra Systems for
    the Ramsey R(3, 8) and R(3, 9) Problems,” arXiv, 2025.
[2] G. Rattan, M. Anders, and S. Brenner, “Satsuma: Structure-Based Symmetry Breaking in SAT,” in Proc. 27th Int. Conf.
    Theory and Applications of Satisfiability Testing (SAT), 2024.
[3] A. Biere, M. Järvisalo, and B. Kiesl, "Handbook of Satisfiability," 2nd ed. IOS Press, 2021.
"""
