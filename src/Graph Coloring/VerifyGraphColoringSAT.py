from pysat.solvers import Solver
from pysat.formula import IDPool
import networkx as nx
import drup

"""
This script demonstrates basic SAT/UNSAT instance verification using SAT solvers.
We apply it to the graph coloring problem.
"""

# Set the number of colors k.
# For k = 4, the instance is UNSAT (not k-colorable).
# For k = 5, the instance is SAT (graph is k-colorable).
k = 4

# Load the Karate Club graph as an adjacency matrix
G= nx.to_numpy_array(nx.karate_club_graph())
colors = [i + 1 for i in range(k)]

# List to hold the clauses for the SAT solver
clauses = []
# Initialize the variable pool, starting from ID 1
vpool = IDPool(start_from=1)

# Whether vertex i-th has color c-th
V = lambda i, c: vpool.id(f'V@{i}@{c}')

# Each vertex must have exactly one color
for i in range(1, len(G) + 1):
    # Each vertex must have at least one color
    clauses.append([V(i, c) for c in colors])

    # Each vertex must have at most one color
    for c1 in colors:
        for c2 in colors:
            if c1 != c2:
                clauses.append([-V(i, c1), -V(i, c2)])

# Two adjacent vertices must have different colors
for i in range(1, len(G) + 1):
    for j in range(1, len(G) + 1):
        if i != j and G[i - 1][j - 1] != 0:
            for c in colors:
                clauses.append([-V(i, c), -V(j, c)])

# Solve the SAT problem with proof logging enabled
with Solver(name='g4', bootstrap_with=clauses, with_proof=True) as solver:
    if solver.solve():
        print("Solver claimed the instance is: SAT")
        model = solver.get_model()

        # Extract positive literals and decode their meaning
        # NOTE: it is also a good practice to save the certificate of SAT instances for future use,
        # although here we do not do that.
        certificate=[vpool.id2obj[m] for m in model if m>0 and m in vpool.id2obj]

        # region: Code for verifying SAT instance
        # NOTE: Verification for problems within NP can be done quickly, yet there might not be a guarantee on some
        # classes beyond it.
        checker_result="VERIFIED"
        try:
            # Map vertex -> assigned color
            colors_assigned_to_vertices = {str(v): set() for v in range(1, len(G) + 1)}
            for assignment in certificate:
                _, vid, color = assignment.split("@")
                colors_assigned_to_vertices[vid].add(color)

            # Check each vertex has exactly one color
            for v in colors_assigned_to_vertices:
                if len(colors_assigned_to_vertices[v]) == 0 or len(colors_assigned_to_vertices[v]) > 1:
                    raise Exception("Each vertex must be assigned exactly one color")
                colors_assigned_to_vertices[v]=list(colors_assigned_to_vertices[v])[0]
            # Check adjacent vertices have different colors
            for v1 in colors_assigned_to_vertices:
                for v2 in colors_assigned_to_vertices:
                    if v1!=v2 and G[int(v1)-1][int(v2)-1]!=0 and colors_assigned_to_vertices[v1]==colors_assigned_to_vertices[v2]:
                        raise Exception("Adjacent vertices must have different colors")
        except Exception as e:
            #print(e)
            checker_result="NOT VERIFIED"
        #endregion
    else:
        print("Solver claimed the instance is: UNSAT")
        # Read and process the DRUP proof
        proof = []
        for line in solver.get_proof():
            tokens = line.strip().split()
            if not tokens:
                continue
            if tokens[0] == 'd':
                literals = list(map(int, tokens[1:]))  # 'd' indicates deletion
            else:
                literals = list(map(int, tokens[:-1]))  # Drop trailing zero
            proof.append(literals)

        # Verify the UNSAT proof using DRUP checker
        # NOTE: it is possible to save the proof to disk for future use,
        # yet DRUP proofs produced by CDCL solvers are often very large.
        # People even in scientific papers usually do not save these proofs,
        # but instead rely on verification with tools like the one below and remove the proofs upon successful
        # verification.
        # Alternatively, concurrent verification is also possible while proof generation occurs,
        # though this is beyond the scope of this basic showcase.
        # Interested readers may look at the paper "Happy ending: An empty hexagon in every set of 30 points"
        # by Marijn JH Heule and Manfred Scheucher, and the repository provided for concurrent verification.
        checker_result=drup.check_proof(clauses, proof, verbose=True)
        checker_result="VERIFIED" if str(checker_result.outcome)=="Outcome.VALID" else "NOT VERIFIED"

    print("And the claim is successfully VERIFIED" if checker_result == "VERIFIED" else "But the claim is NOT VERIFIED")
