from pysat.formula import *
from pysat.formula import IDPool
from pysat.solvers import Solver
from itertools import combinations
from pysat.card import CardEnc,EncType

"""
This script is divided into two parts. In the first part, I aim to develop a formula that can answer simple questions
about the Tic-Tac-Toe game, such as whether it is possible to win under certain assumptions.

The goal is to provide an understanding of how turn-based adversarial games can be encoded into SAT. Additionally,
the aim is to introduce the concept of incorporating assumptions into incremental SAT solving.

The second part is optional. While SAT solvers can answer basic questions, such as whether it is possible to win or
draw under certain assumptions, the situation becomes more complex when dealing with advanced queries. For example, 
determining whether there is a winning strategy for a player to force a victory.

For these types of queries, Quantified Boolean Formulas (QBF) offer a potential solution. QBF extends Boolean logic by
introducing existential and universal quantifiers. One approach is to modify the encoding provided in Part 1 to include
alternating quantification, though some additional adjustments may be necessary. Once that’s done, a QBF solver can be 
used to answer more sophisticated questions.

Since this part is optional and the concepts involved are quite deep, I’ve chosen to use a built-in tool for simplicity.
As an example, I demonstrate how to use a recently proposed tool that can automatically encode such games into QBF.

Finally, I’ve included some suggested materials at the end for further reading.

"""

# Initialize CNF (Conjunctive Normal Form) and ID pool for variable management
cnf = CNF()
vpool = IDPool(start_from=1)

# Possible values for the cells on the Tic-Tac-Toe board
values = ["X", "O", "-"]

# Helper function to map a cell (i, j) at time t to a unique variable representing its value
C = lambda i, j, t, v: vpool.id(f"C@{i}@{j}@{t}@{v}")  # Cell value at (i,j) at time t is v

# Loop over each time step, and for each cell (i,j), enforce the constraints that:
# There must be exactly one value for each cell at each time t (i.e., cell must be either 'X', 'O', or '-').
for t in range(10):
    for i in range(3):
        for j in range(3):
            # At least one value (cell must be either X, O, or -)
            cnf.append([C(i, j, t, v) for v in values])
            # At most one value (cell can't have more than one value at a time)
            for v1, v2 in combinations(values, 2):
                cnf.append([-C(i, j, t, v1), -C(i, j, t, v2)])

# Enforce that 'X' or 'O' remains fixed across time steps (i.e., once a player places a mark, it stays there)
for t in range(9): # For each time step except the last
    for i in range(3):
        for j in range(3):
            for v in ["X", "O"]:
                # If a cell is filled with 'X' or 'O' at time t, it must also be filled with the same value at time t+1
                cnf.append([-C(i, j, t, v), C(i, j, t + 1, v)])

# Initial board configuration: all cells are empty at time 0
cnf.extend(CardEnc.equals([C(i, j, 0, "-") for i in range(3) for j in range(3)], bound=9, vpool=vpool, encoding=EncType.seqcounter))

# Define a move variable M(i, j, t) that tracks whether a cell (i,j) was filled at time t
M = lambda i, j, t: vpool.id(f"M@{i}@{j}@{t}")

# Enforce move constraints:
# At most one move should be made at each time step
for t in range(9):
    # Ensure that at most one move is made at any given time
    move_vars = [M(i, j, t) for i in range(3) for j in range(3)]
    cnf.extend(CardEnc.atmost(move_vars, bound=1, vpool=vpool, encoding=EncType.seqcounter))

    # Enforce the relationship between move and cell value:
    # If a move is made at time t, the cell must be empty at time t and filled at time t+1.
    for i in range(3):
        for j in range(3):
            # If M(i,j,t) is true, then the cell (i,j) must be empty at time t and filled with 'X' or 'O' at time t+1
            cnf.append([-M(i, j, t), C(i, j, t, "-")])
            cnf.append([-M(i, j, t), -C(i, j, t + 1, "-")])
            # If the cell is filled with 'X' or 'O' at time t, the move must have happened
            cnf.append([-C(i, j, t, "-"), C(i, j, t + 1, "-"), M(i, j, t)])
            # Enforce that the move at time t corresponds to 'X' if t is even, 'O' if t is odd
            cnf.append([-M(i, j, t), C(i, j, t + 1, "X" if t%2==0 else "O")])

# After time t=9, no more moves are allowed
for i in range(3):
    for j in range(3):
        cnf.append([-M(i, j, 9)])



# Winning conditions for rows, columns, and diagonals (R0, R1, R2, C0, C1, C2, D, d)
conditions=['R0','R1','R2','C0','C1','C2','D','d']
WCond = lambda c,p,t: vpool.id(f"WCond@{c}@{p}@{t}")# Winning condition for player p at time t for condition c

# Loop over time steps and for each player ('X' or 'O'), enforce the winning conditions (rows, columns, diagonals)
for t in range(10):
    for p in ["X","O"]:
        for i in range(3):
            # Enforce the row winning condition: WCond("R0", "X", t) <-> C(i, 0, t, "X") and C(i, 1, t, "X") and C(i, 2, t, "X")
            cnf.append([-WCond(f"R{i}", p, t), C(i, 0, t, p)])
            cnf.append([-WCond(f"R{i}", p, t), C(i, 1, t, p)])
            cnf.append([-WCond(f"R{i}", p, t), C(i, 2, t, p)])
            cnf.append([-C(i, 0, t, p), -C(i, 1, t, p), -C(i, 2, t, p), WCond(f"R{i}", p, t)])


            # Enforce the column winning condition: WCond("C0", "X", t) <-> C(0, i, t, "X") and C(1, i, t, "X") and C(2, i, t, "X")
            cnf.append([-WCond(f"C{i}", p, t), C(0, i, t, p)])
            cnf.append([-WCond(f"C{i}", p, t), C(1, i, t, p)])
            cnf.append([-WCond(f"C{i}", p, t), C(2, i, t, p)])
            cnf.append([-C(0, i, t, p), -C(1, i, t, p), -C(2, i, t, p), WCond(f"C{i}", p, t)])

        # Enforce diagonal winning condition (top-left to bottom-right)
        cnf.append([-WCond(f"D", p, t), C(0, 0, t, p)])
        cnf.append([-WCond(f"D", p, t), C(1, 1, t, p)])
        cnf.append([-WCond(f"D", p, t), C(2, 2, t, p)])
        cnf.append([-C(0, 0, t, p), -C(1, 1, t, p), -C(2, 2, t, p), WCond(f"D", p, t)])

        # Enforce diagonal winning condition (top-right to bottom-left)
        cnf.append([-WCond(f"d", p, t), C(2, 0, t, p)])
        cnf.append([-WCond(f"d", p, t), C(1, 1, t, p)])
        cnf.append([-WCond(f"d", p, t), C(0, 2, t, p)])
        cnf.append([-C(2, 0, t, p), -C(1, 1, t, p), -C(0, 2, t, p), WCond(f"d", p, t)])

# Define whether a player has already won at time t
W = lambda p,t: vpool.id(f"W@{p}@{t}")

# Enforce that if a player has won at time t, one of the winning conditions must be true
for t in range(10):
    for p in ["X","O"]:
        # If W(p,t) is true, then one of the winning conditions must be true for that player
        cnf.append([-W(p,t)]+[WCond(c,p,t) for c in conditions])
        # If WCond(c, p, t) is true, then W(p,t) must be true
        for c in conditions:
             cnf.append([-WCond(c,p,t),W(p,t)])

# Enforce that if either player wins, no further moves can occur
for t in range(10):
    for i in range(3):
        for j in range(3):
            # If X wins, no moves can happen at (i, j)
            cnf.append([-W("X",t),-M(i,j,t)])
            # If O wins, no moves can happen at (i, j)
            cnf.append([-W("O",t),-M(i,j,t)])

# If neither player wins by time t, at least one move must be made
for t in range(9):
    cnf.append([W("X",t),W("O",t)]+[M(i,j,t) for i in range(3) for j in range(3)])

# Define the results (who won or if it's a draw)
results= {"X","O","D"}
R = lambda r: vpool.id(f"R@{r}")
# Ensure that the results are consistent (either X, O, or a draw)
cnf.append([R(r) for r in results])
for r1,r2 in combinations(results,2):cnf.append([-R(r1),-R(r2)])

# Enforce that R(p) is true if and only if player p has won at some time step t (0 <= t <= 9)
for p in ["X","O"]:
    # R(p) is true, then W(p, t) must be true for at least one t in range(10)
    cnf.append([-R(p)] + [W(p, t) for t in range(10)])

    # For each time step t, ensure that if W(p, t) is true, then R(p) must be true
    for t in range(10):
        cnf.append([-W(p, t), R(p)])



with Solver(name='Cadical195', bootstrap_with=cnf, use_timer=True) as solver:
    """
        Assumptions are unit clauses, meaning assumptions like [1, 2, -4] are interpreted as [[1], [2], [-4]].
        Assumptions can be used in incremental SAT solving, allowing multiple queries on the same underlying formula.
        The advantage is that the solver can benefit from the information gained during previous calls,
        improving performance for subsequent queries.
    """
    basic_queries=[([R("X")], "Is it possible for player X to win the game?"),
                   ([R("D")],"Could the game end in a draw?"),
                   ([R("O")],"Is it possible for player O to win the game?"),
                   ([W("X",5),C(1,2,2,"O")]+[-C(1,1,t,"X") for t in range(6)],
                    "Is it possible for player X to win by time 5, if player O takes cell 1,2 and player X is prohibited from playing 1,1?")]

    for assumptions,query_text in basic_queries:
        print(query_text)
        if solver.solve(assumptions=assumptions):
            print("SAT")
            model = solver.get_model()
            output = {t: [[None] * 3, [None] * 3, [None] * 3] for t in range(10)}
            moves = [None] * 10
            status = ["Finished"] * 10
            WinByState = [False] * 10
            Res = ""
            for m in model:
                if m > 0 and m in vpool.id2obj:
                    item = vpool.id2obj[m].split('@')
                    if item[0] == "C":
                        i, j, t, v = int(item[1]), int(item[2]), int(item[3]), item[4]
                        output[t][i][j] = v
                    if item[0] == "M":
                        i, j, t = int(item[1]), int(item[2]), int(item[3])
                        moves[t] = (i, j)
                    if item[0] == "S":
                        status[int(item[1])] = "Continue"
                    if item[0] == "W":
                        p, t = item[1], int(item[2])
                        WinByState[t] = p
                    if item[0] == "R":
                        Res = item[1]

            for t in range(10):
                print(moves[t])
                for r in range(3):
                    print(output[t][r])
                    print()
                print("-" * 15)
            print(Res)

        else:
            print("UNSAT")



"""
                                                  (OPTIONAL)
                                                    PART 2.

SAT solvers are effective for straightforward planning tasks. However, in adversarial scenarios like positional games
(e.g., Tic-Tac-Toe), although SAT solvers can easily answer simple queries (e.g., whether winning the game is possible
or how many ways the game might end in a draw), they are not well-suited for more sophisticated questions. For example,
determining whether there is a forcing winning strategy for one player is a much more complex query.

One solution to this problem is to use Quantified Boolean Formulas (QBFs) and their corresponding solvers. QBFs are
propositional logical formulas that include both existential and universal quantifiers. This allows us to model
alternating moves in adversarial games.

For example, in a game like Tic-Tac-Toe, we can represent the sequence of moves as follows:

Does there exist a move for Player 1 at time 1,
 such that for all possible moves Player 2 could make at time 2,
 there exists a subsequent move for Player 1,
  such that for all possible moves Player 2 could make,
   and so on, the set of logical clauses encoding the game's rules and constraints is satisfied
    (e.g., after a maximum of 9 moves in Tic-Tac-Toe)?

[The above example is inspired by the one presented here:
                                                    https://www.cs.cornell.edu/~carlos/webpage/real-cornell_april.PDF]

The formula is satisfiable (SAT) if there exists a strategy for Player 1 that guarantees a win. If it is unsatisfiable
(UNSAT), then Player 1 does not have a forcing strategy that leads to a win.

----------------------------------------------------------------------------------------------------------------------

Positional games are one of the applications of QBFs. The paper Positional Games and QBF: A Polished Encoding by
Valentin Mayer-Eichberger and Abdallah Saffidine proposes encoders that can be used to solve such games using the
following procedure:
1. Specify the format of the game (e.g., timesteps, which player moves at each step, initial positions,
        winning conditions, etc.) in a .pg file format.
2. Pass the .pg file to the tool provided in the paper, which encodes the problem in QBF and outputs a .bule file.
3. Process the .bule file to generate a .qdimacs file, which can be solved by a QBF solver.

To adopt this, the following tools are needed:
1. Positional Games QBF Encoding – This tool generates the .bule file from the .pg game description.
2. Bule File Processor – This tool converts the .bule file to a .qdimacs file.
3. DepQBF – A QBF solver that processes the .qdimacs file and solves the instance.

Following this procedure, we can use QBF solvers to analyze positional games and determine if a player has a forcing
winning strategy. The following provides the details on how to do so.

----------------------------------------------------------------------------------------------------------------------

Setting up and Running the Positional Games QBF Encoding

1. Clone the repository for positional games QBF encoding:
git clone https://github.com/vale1410/positional-games-qbf-encoding.git
cd positional-games-qbf-encoding

2. Install Go (if not already installed):
For Snap users: sudo snap install go
For APT users: sudo apt install golang-go

3. Build the encoder: Once Go is installed, build the encode.go file: go build encode.go
After successful compilation, the encode executable should be available. You can check it by running: ./encode -h

4. Create a game file: Now, create a file representing the game you wish to solve. For example,
here’s a Tic-Tac-Toe example (TicTacToe.pg):
#times
t1 t2 t3 t4 t5 t6 t7 t8 t9
#blackturns
t1 t3 t5 t7 t9
#positions
a1 a2 a3 b1 b2 b3 c1 c2 c3
#blackwins
a1 b1 c1
a2 b2 c2
a3 b3 c3
a1 a2 a3
b1 b2 b3
c1 c2 c3
a1 b2 c3
a3 b2 c1
#whitewins
a1 b1 c1
a2 b2 c2
a3 b3 c3
a1 a2 a3
b1 b2 b3
c1 c2 c3
a1 b2 c3
a3 b2 c1

5. Next, clone the Bule repository:
git clone https://github.com/vale1410/bule.git
cd bule/src


7. After installing the dependencies, build Bule:
If opam is not installed, use: sudo apt install opam
If dune is not installed, run: sudo apt install dune
opam install menhirLib minisat qbf tsort menhir
eval $(opam env)
dune clean
dune build bin/Bule.exe

The Bule.exe file will be created in _build/default/bin/Bule.exe

(Optional)
Prepare your files: For simplicity, ensure that Bule.exe and your game file (TicTacToe.pg)
are in the same directory as encode from the positional-games-qbf-encoding repository.

8. Create the benchmark script: Create a script make_my_benchmark.sh with the following content:
#!/bin/zsh

# Check if enough arguments are provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <encoding> <input_file> <output_file>"
    exit 1
fi

# Assign the arguments to variables
encoding=$1            # First argument: Encoding (e.g., 243)
input_file=$2          # Second argument: Input file (e.g., gttt-3x3.gpg)
output_file=$3         # Third argument: Output file (e.g., result.qdimacs)

# Build the encoder (ensure Go is installed)
go build encode.go

# Encode the input file
temp_file=$(mktemp)
./encode $input_file --enc=$encoding > $temp_file

# Run Bule.exe with the encoding and the encoded file
start=$(date +%s%3N)
./Bule.exe bule/pg$encoding.bul --output=qdimacs $temp_file > $output_file
end=$(date +%s%3N)

# Output the processing time and file info
echo "Processed $input_file in $((end-start)) ms"

9. Run the benchmark: Now, run the benchmark with the following command:
./make_my_benchmark.sh 243 TicTacToe.pg TicTacToe.qdimacs

10. Install a QBF solver: For example, install depqbf: sudo apt install depqbf
Check the solver: depqbf -h

11. Run the QBF solver on your .qdimacs file:
depqbf TicTacToe.qdimacs

If the solver returns UNSAT, it means there is no winning strategy where the black player can force the opponent to
lose. In other words, no matter how the game is played, the black player cannot guarantee a victory under the given
conditions.
On the other hand, if the solver returns SAT, it indicates that there is indeed a winning strategy for the black player
to force the opponent to lose.
For the Tic-Tac-Toe, the result is UNSAT, meaning there is no guaranteed winning strategy for black ('X' player) under
the specified settings.
"""

"""
Some resources that interested readers might want to check out:

- Reasoning with Quantified Boolean Formulas by Marijn J.H. Heule: Automated Reasoning and Satisfiability, 2020
- Solving Two-Player Games: https://www.cs.cornell.edu/~carlos/webpage/real-cornell_april.PDF
- PyQBF: A Python Framework for Solving Quantified Boolean Formulas by Mark Peyrer, Maximilian Heisinger, Martina Seidl
- Positional Games and QBF: A Polished Encoding by Valentin Mayer-Eichberger, Abdallah Saffidine
- Handbook of Satisfiability (2nd edition) edited by Armin Biere, Marijn Heule, and Hans van Maaren: Chapters dedicated
    to QBF
"""