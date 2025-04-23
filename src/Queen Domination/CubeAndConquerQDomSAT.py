from plumbum import local
from pysat.formula import CNF
from pysat.solvers import Solver
import os
import uuid
from pysat.card import CardEnc, EncType
from multiprocessing import Pool

"""
This script demonstrates a basic adaptation of parallel/distributed SAT solving.

The core idea is to divide the search space into non-overlapping subspaces and solve each in parallel.

We adopt the Cube-and-Conquer approach [1], where the original SAT formula is partitioned into smaller sub-problems 
called cubes. Each cube represents a partial assignment (a conjunction of literals) that constrains the solution space.

Workflow:
1. Use a look-ahead solver (e.g., `march_cu`) to partition the formula into cubes.
2. For each cube, spawn a parallel SAT solver process with the cube added as unit clauses.
3. If any solver returns SAT, terminate the rest and return the satisfying assignment.
4. If all solvers return UNSAT, then the original formula is UNSAT.

Additional Notes:
- This prototype is written in Python for clarity. In practice, shell scripts and more optimized infrastructure are 
    often preferred.
- For simplicity, we omit detailed logging, cube/formula serialization, or runtime statistics, though these can be 
    easily integrated.
- In safety-critical or formally verified applications, results are often checked by an additional verification phase.
- Although this script runs everything on a single multi-core machine, the same principles extend naturally to 
    distributed or cloud-based environments for large-scale SAT instances.
"""


"""You can replace this encoding with any other SAT encoding of interest—the rest of the pipeline remains unchanged."""
def queen_dom_to_SAT(n, gamma, enc_type_atmost=EncType.seqcounter, enc_type_atleast=EncType.seqcounter):
    """
        Encodes the n x n Queen Domination problem as a SAT instance with at most `gamma` queens on the board.
        Returns: A list of CNF clauses representing the problem constraints.
    """
    get_top_id= lambda clauses:max([max([abs(item) for item in c]) for c in clauses])

    clauses = []
    # To ensure compatibility with SAT solvers, indices start from 1 and end at n^2
    V = [i + 1 for i in range(n * n)]
    top_id = V[-1]

    for i in range(len(V)):
        N = [V[i]]
        r, c = i // n, i % n
        # Squares in the same row
        N += V[r * n:r * n + n]
        # Squares in the same column
        N += V[c::n]
        # Squares in the same diagonal (from top left to bottom right)
        N += [V[j] for j in range(len(V)) if r - c == ((j // n) - (j % n))]
        # Squares in the same diagonal (from top right to bottom left)
        N += [V[j] for j in range(len(V)) if r + c == ((j // n) + (j % n))]
        N = list(set(N))
        clauses.append(N)

    clauses += CardEnc.atmost(lits=V, top_id=top_id, bound=gamma, encoding=enc_type_atmost).clauses

    if enc_type_atleast is not None:
        top_id = get_top_id(clauses)
        clauses += CardEnc.atleast(lits=V, top_id=top_id, bound=gamma, encoding=enc_type_atleast).clauses

    # OR
    # clauses += CardEnc.equals(lits=[-v for v in V], top_id=top_id, bound=((n*n)-gamma), encoding=enc_type_atmost).clauses

    return clauses

def solve_wrapper(args):
    """
    Worker function to solve a SAT problem under a given cube (partial assignment).
    """
    base_clauses, cube = args
    cnf = CNF(from_clauses=base_clauses)
    for lit in cube:
        cnf.append([lit])# Add cube as unit clauses
    with Solver(bootstrap_with=cnf) as solver:
        if solver.solve():
            return ("SAT", solver.get_model())
        else:
            return ("UNSAT", None)

def parallel_pysat_solver_with_cubes(base_clauses, cubes, max_workers=None):
    """
    Solves the SAT problem in parallel by assigning each cube to a separate process.
    Returns a satisfying model if found; otherwise, None.
    """
    with Pool(processes=max_workers) as pool:
        for status, data in pool.imap_unordered(solve_wrapper, [(base_clauses, cube) for cube in cubes]):
            if status == "SAT":
                pool.terminate()
                return data  # model
    return None

def load_cubes(fpath):
    """
    Loads cubes from a file in `march_cu` output format.
    """
    cubes = []
    with open(fpath, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('a '):
                literals = list(map(int, line.split()[1:]))
                if literals[-1] == 0:
                    literals = literals[:-1]
                cubes.append(literals)
    return cubes


if __name__ == "__main__":
    tmp_input_file = f"formula_{uuid.uuid4().hex}.cnf"
    tmp_output_file = f"cubes_{uuid.uuid4().hex}"
    try:
        # Example configurations:
        # SAT: n, gamma = 8, 5
        # UNSAT: n, gamma = 8, 4
        n,gamma = 8,4
        clauses=queen_dom_to_SAT(n, gamma)
        cnf = CNF(from_clauses=clauses)

        cnf.to_file(tmp_input_file)
        cnf.to_file(tmp_input_file)

        # Generate cubes using external look-ahead solver
        march_cu = local["./march_cu"]
        march_cu(tmp_input_file, "-o", tmp_output_file)
        cubes = load_cubes(tmp_output_file)
        print(f"# Cubes generated: {len(cubes)}")

        result = parallel_pysat_solver_with_cubes(clauses, cubes, max(1, os.cpu_count() - 1))

        if result:
            print("SAT found with model:")
            print(result)
        else:
            print("All cubes returned UNSAT. Problem is UNSAT.")

    except Exception as e:
        print("Error:", e)
    finally:
        if os.path.exists(tmp_input_file):
            os.remove(tmp_input_file)
        if os.path.exists(tmp_output_file):
            os.remove(tmp_output_file)

"""
References:
[1] M. Heule, O. Kullmann, S. Wieringa, and A. Biere, "Cube and Conquer: Guiding CDCL SAT Solvers by Lookaheads,"
    In Haifa Verification Conference, Springer, 2011.
"""