import pygame
import time
from pysat.solvers import Solver
from pysat.formula import *

"""
NOTE: hard coded
"""

"""
To use the graphical version, you must provide images and sounds with the specific names mentioned in the code.
For example, '8-bit-move-001-171817.mp3'. You can use images and sounds of your choice, but they must have these
exact names. Alternatively, you can modify the code to match your images and sounds' paths and filenames.
"""

manual_mode = True

"""
In this game, the agent has limited knowledge. It only knows about the squares it has visited and not the
ones it hasn’t explored yet. For example, at the start of the game, the agent doesn't know where the 
Wumpus or the pit are located. It only gains this knowledge based on the current location and the squares
it has visited. It must rely on its exploration and the actions it takes to gather information.
However, for simplicity, I’m displaying the entire environment here.

To play the game, you can use the following controls:

d: Turn east

a: Turn west

w: Turn north

s: Turn south

Enter: Shoot

Space: Move forward

Esc: Climb (exit the cave)

g: Grab (pick up items)

The agent’s movement and decisions are based on what it can see in its current square and what it has
learned from the squares it has visited.
"""

def sleep_for_agent_mode():
    time.sleep(0.7)  # Adjust the sleep duration for the desired pace


class WumpusWorldQueryDatabase:

    def __init__(self, grid_size=4):
        self.grid_size = grid_size

        self.is_wumpus_alive = True
        self.has_arrow = True

        self.unvisited_cells = set(
            [(x, y) for x in range(self.grid_size) for y in range(self.grid_size) if (x, y) != (0, 0)]
        )
        self.visited_cells={(0,0)}

        self.variable_pool = IDPool(start_from=1)

        for x in range(grid_size):
            for y in range(grid_size):
                self.P(x, y)
                self.W(x, y)
                self.B(x, y)
                self.S(x, y)

        self.knowledge_base = self.initialize_knowledge_base()
        self.solver=Solver(name='Cadical195', bootstrap_with=CNF(from_clauses=self.knowledge_base))

    def get_adjacent_cells(self, x, y):
        return [
            (x + dx, y + dy) for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
            if 0 <= x + dx < self.grid_size and 0 <= y + dy < self.grid_size
        ]


    # Px,y is true if there is a pit in [x, y].
    def P(self, x, y):
        return self.variable_pool.id(f'P@{x}@{y}')

    #Wx,y is true if there is a wumpus in [x, y], dead or alive.
    def W(self, x, y):
        return self.variable_pool.id(f'W@{x}@{y}')


    #Bx,y is true if there is a breeze in [x, y].
    def B(self, x, y):
        return self.variable_pool.id(f'B@{x}@{y}')

    #Sx,y is true if there is a stench in [x, y].

    def S(self, x, y):
        return self.variable_pool.id(f'S@{x}@{y}')

    def initialize_knowledge_base(self):
        clauses = []

        # Starting square does not contain a pit or a wumpus
        clauses.append([-self.P(0, 0)])
        clauses.append([-self.W(0, 0)])

        # There is exactly one Wumpus
        clauses.append([self.W(x, y) for x in range(self.grid_size) for y in range(self.grid_size)])

        for x1 in range(self.grid_size):
            for y1 in range(self.grid_size):
                for x2 in range(self.grid_size):
                    for y2 in range(self.grid_size):
                        if (x1, y1) != (x2, y2):
                            clauses.append([-self.W(x1, y1), -self.W(x2, y2)])

        # Breeze logic: a square is breezy if and only if a neighboring square has a pit
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                neighbors = self.get_adjacent_cells(x, y)

                clauses.append([-self.B(x, y)] + [self.P(nx, ny) for nx, ny in neighbors])

                for nx, ny in neighbors:
                    clauses.append([-self.P(x, y), self.B(nx, ny)])

        # Stench logic: a square is smelly if and only if a neighboring square has a wumpus
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                neighbors = self.get_adjacent_cells(x, y)

                clauses.append([-self.S(x, y)] + [self.W(nx, ny) for nx, ny in neighbors])

                for nx, ny in neighbors:
                    clauses.append([-self.W(x, y), self.S(nx, ny)])

        return clauses

    def update_knowledge_base_with_percept(self, percept):
        agent_x, agent_y = percept['AgentLoc']

        if (agent_x, agent_y) in self.unvisited_cells:
            self.unvisited_cells.remove((agent_x, agent_y))
            self.visited_cells.add((agent_x, agent_y))
            self.solver.add_clause([-self.P(agent_x, agent_y)])

        if percept['Bang'] and self.has_arrow:
            self.has_arrow = False

        self.solver.add_clause([
            self.S(agent_x, agent_y) if percept['Stench'] else -self.S(agent_x, agent_y)
        ])

        self.solver.add_clause([
            self.B(agent_x, agent_y) if percept['Breeze'] else -self.B(agent_x, agent_y)
        ])

        if percept['Scream']:
            self.is_wumpus_alive = False
        else:
            if self.is_wumpus_alive:
                self.solver.add_clause([-self.W(agent_x, agent_y)])

    # Check if the agent can infer with certainty that a Wumpus is present at (x, y)
    def has_wumpus(self, x, y):
        return not self.solver.solve(assumptions=[-self.W(x, y)])

    # Check if the agent can infer with certainty that a Wumpus is not present at (x, y)
    def has_no_wumpus(self, x, y):
        return not self.solver.solve(assumptions=[self.W(x, y)])

    # Check if the agent can infer with certainty that a pit is present at (x, y)
    def has_pit(self, x, y):
        return not self.solver.solve(assumptions=[-self.P(x, y)])

    # Check if the agent can infer with certainty that a pit is not present at (x, y)
    def has_no_pit(self, x, y):
        return not self.solver.solve(assumptions=[self.P(x, y)])

    # Check if the agent can infer with certainty that the cell (x, y) is safe
    def is_safe(self, x, y):
        if not self.has_no_pit(x, y):
            return False

        if not self.is_wumpus_alive:
            return True

        return self.has_no_wumpus(x, y)

    def get_safe_cells(self):
        return {(x, y) for x in range(self.grid_size) for y in range(self.grid_size) if self.is_safe(x, y)}

    def get_unvisited_cells(self):
        return self.unvisited_cells

    def get_visited_cells(self):
        return self.visited_cells

    def get_possible_wumpus_locations(self):
        return {
            (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
            if not self.has_no_wumpus(x, y)
        }




# Note: Instead of invoking the SAT solver from scratch each time, it is worth exploring slight code modifications 
# to leverage incremental SAT solving.
def PlanShot(current_face, current_loc, goals_locs, safe_squares, n=4, T=100):
    def PlanShot_(current_face, current_loc, goals_locs, safe_squares, n=4, T=10):
        """
            Plans the agent's movement and shooting actions to reach a goal location within a given time frame.

            Parameters:
            - current_face (str): The current direction the agent is facing. Possible values:
             ['FacingWest', 'FacingNorth', 'FacingEast', 'FacingSouth']
            - current_loc (tuple): The current (x, y) location of the agent.
            - goals_locs (set of tuples): Set of target locations (x, y) the agent aims to reach.
            - safe_squares (set of tuples): Set of known safe locations where the agent can navigate.
            - n (int): Grid size (nxn) representing the environment's dimensions. Default is 4.
            - T (int): Maximum number of time steps (or actions) allowed to reach the goal. Default is 10.

            Returns:
            - Returns a plan to reach one of the goals, or None if no plan is found.
        """

        # Exclude the current location from the list of goal locations (agent is already there).
        goals_locs=goals_locs.difference({current_loc})
        # If no goals remain, exit early.
        if len(goals_locs)==0: return None

        # Initialize an ID pool to generate unique identifiers for logical variables.
        vpool = IDPool(start_from=1)

        # Set of all possible squares in the grid.
        all_squares = set([(i, j) for i in range(n) for j in range(n)])

        # Boolean variable indicating whether the agent can shoot from (src_x, src_y) to (goal_x, goal_y) at T-1.
        Shoot = lambda src_x, src_y, goal_x, goal_y: vpool.id(f'Shoot@{src_x}@{src_y}@{goal_x}@{goal_y}')

        # Movement and rotation decision variables for each time step t.
        Forward = lambda t: vpool.id(f'Forward@{t}')
        TurnEast = lambda t: vpool.id(f'TurnEast@{t}')
        TurnWest = lambda t: vpool.id(f'TurnWest@{t}')
        TurnNorth = lambda t: vpool.id(f'TurnNorth@{t}')
        TurnSouth = lambda t: vpool.id(f'TurnSouth@{t}')

        # Boolean variable indicating if the agent is at location (x, y) at time t.
        L = lambda x, y, t: vpool.id(f'L@{x}@{y}@{t}')

        # Directional variables representing the agent's orientation at time t.
        FacingWest = lambda t: vpool.id(f'FacingWest@{t}')
        FacingNorth = lambda t: vpool.id(f'FacingNorth@{t}')
        FacingEast = lambda t: vpool.id(f'FacingEast@{t}')
        FacingSouth = lambda t: vpool.id(f'FacingSouth@{t}')

        # Initialize variables for all time steps and grid positions.
        for t in range(T):
            Forward(t)
            TurnEast(t)
            TurnWest(t)
            TurnNorth(t)
            TurnSouth(t)

            FacingWest(t)
            FacingNorth(t)
            FacingEast(t)
            FacingSouth(t)

            for x in range(n):
                for y in range(n):
                    for t in range(T):
                        L(x, y, t)

        # Create all possible shooting variables from any square to goal locations.
        all_shoots = []
        for glocX, glocY in goals_locs:
            for src_x in range(n):
                for src_y in range(n):
                    if (glocX, glocY) != (src_x, src_y):
                        all_shoots.append(Shoot(src_x, src_y, glocX, glocY))

        clauses = []

        # Ensure at least one shooting action is considered.
        clauses.append(all_shoots)

        # Map initial facing direction to its corresponding variable.
        if current_face == "FacingEast":
            current_face_ = FacingEast(0)
        elif current_face == "FacingWest":
            current_face_ = FacingWest(0)
        elif current_face == "FacingNorth":
            current_face_ = FacingNorth(0)
        elif current_face == "FacingSouth":
            current_face_ = FacingSouth(0)

        # Prevent the agent from entering unsafe squares at any time step.
        not_safe = all_squares.difference(safe_squares)
        for ns in not_safe:
            for t in range(T):
                clauses.append([-L(ns[0], ns[1], t)])

        # Ensure exactly one orientation is valid at each time step.
        Orientations = [FacingWest, FacingNorth, FacingEast, FacingSouth]
        for t in range(T):
            clauses.append([o(t) for o in Orientations])# At least one direction must hold.
            for o1 in range(len(Orientations)):
                for o2 in range(o1 + 1, len(Orientations)):
                    clauses.append([-Orientations[o1](t), -Orientations[o2](t)]) # No two directions simultaneously.

        # Ensure the agent occupies exactly one location at each time step.
        for t in range(T):
            clauses.append([L(x, y, t) for x in range(n) for y in range(n)])
            for x1 in range(n):
                for y1 in range(n):
                    for x2 in range(n):
                        for y2 in range(n):
                            if (x1, y1) != (x2, y2):
                                clauses.append([-L(x1, y1, t), -L(x2, y2, t)])

        # Ensure the agent performs exactly one action per time step, except for the last step.
        # In the final step, the agent must shoot regardless.
        Actions = [Forward, TurnEast, TurnWest, TurnNorth, TurnSouth]
        for t in range(T - 1):
            # The agent must perform at least one action at each time step.
            clauses.append([o(t) for o in Actions])
            for o1 in range(len(Actions)):
                for o2 in range(o1 + 1, len(Actions)):
                    # Ensure no two actions are performed simultaneously in the same time step.
                    clauses.append([-Actions[o1](t), -Actions[o2](t)])

        # In the final time step, no actions should be allowed.
        # In the final step, the agent must shoot
        for action in Actions:
            clauses.append([-action(T - 1)])

        # For each goal location, set up constraints on the agent's ability to shoot based on its position and direction.
        for glocX, glocY in goals_locs:
            for src_x in range(n):
                for src_y in range(n):
                    if (glocX, glocY) != (src_x, src_y):
                        # Depending on the relative position to the goal, the agent must be facing the goal direction to shoot.
                        if src_x == glocX and src_y < glocY:
                            clauses.append([L(src_x, src_y, T - 1), -Shoot(src_x, src_y, glocX, glocY)])
                            clauses.append([FacingEast(T - 1), -Shoot(src_x, src_y, glocX, glocY)])
                        elif src_x == glocX and src_y > glocY:
                            clauses.append([L(src_x, src_y, T - 1), -Shoot(src_x, src_y, glocX, glocY)])
                            clauses.append([FacingWest(T - 1), -Shoot(src_x, src_y, glocX, glocY)])
                        elif src_x < glocX and src_y == glocY:
                            clauses.append([L(src_x, src_y, T - 1), -Shoot(src_x, src_y, glocX, glocY)])
                            clauses.append([FacingSouth(T - 1), -Shoot(src_x, src_y, glocX, glocY)])
                        elif src_x > glocX and src_y == glocY:
                            clauses.append([L(src_x, src_y, T - 1), -Shoot(src_x, src_y, glocX, glocY)])
                            clauses.append([FacingNorth(T - 1), -Shoot(src_x, src_y, glocX, glocY)])
                        else:
                            clauses.append([-Shoot(src_x, src_y, glocX, glocY)])

        # Ensure that actions such as turning and moving have the expected consequences.
        for t in range(T - 1):

            # For each direction, ensure that the agent faces the correct direction after a turn.
            clauses.append([-FacingWest(t), TurnEast(t), TurnNorth(t), TurnSouth(t), FacingWest(t + 1)])
            clauses.append([-FacingEast(t), TurnWest(t), TurnNorth(t), TurnSouth(t), FacingEast(t + 1)])
            clauses.append([-FacingNorth(t), TurnEast(t), TurnWest(t), TurnSouth(t), FacingNorth(t + 1)])
            clauses.append([-FacingSouth(t), TurnEast(t), TurnWest(t), TurnNorth(t), FacingSouth(t + 1)])

            # Ensure that the agent faces the correct direction when turning.
            clauses.append([-FacingWest(t + 1), TurnWest(t), FacingWest(t)])
            clauses.append([-FacingEast(t + 1), TurnEast(t), FacingEast(t)])
            clauses.append([-FacingNorth(t + 1), TurnNorth(t), FacingNorth(t)])
            clauses.append([-FacingSouth(t + 1), TurnSouth(t), FacingSouth(t)])

            # For each grid position, define the consequences of moving forward.
            for x in range(n):
                for y in range(n):
                    # Handling movement based on the current facing direction (West, East, North, South).

                    # Forward
                    # FacingWest
                    if y - 1 >= 0:
                        clauses.append([L(x, y - 1, t + 1), -L(x, y, t), -FacingWest(t), -Forward(t)])
                    else:
                        clauses.append([-L(x, y, t), -FacingWest(t), -Forward(t), L(x, y, t + 1)])
                        clauses.append([-L(x, y, t), -FacingWest(t), -Forward(t)])
                    # FacingEast
                    if y + 1 < n:
                        clauses.append([L(x, y + 1, t + 1), -L(x, y, t), -FacingEast(t), -Forward(t)])
                    else:
                        clauses.append([-L(x, y, t), -FacingEast(t), -Forward(t), L(x, y, t + 1)])
                        clauses.append([-L(x, y, t), -FacingEast(t), -Forward(t)])
                    # FacingNorth
                    if x - 1 >= 0:
                        clauses.append([L(x - 1, y, t + 1), -L(x, y, t), -FacingNorth(t), -Forward(t)])
                    else:
                        clauses.append([-L(x, y, t), -FacingNorth(t), -Forward(t), L(x, y, t + 1)])
                        clauses.append([-L(x, y, t), -FacingNorth(t), -Forward(t)])
                    # FacingSouth
                    if x + 1 < n:
                        clauses.append([L(x + 1, y, t + 1), -L(x, y, t), -FacingSouth(t), -Forward(t)])
                    else:
                        clauses.append([-L(x, y, t), -FacingSouth(t), -Forward(t), L(x, y, t + 1)])
                        clauses.append([-L(x, y, t), -FacingSouth(t), -Forward(t)])

                    # (L(x,y,t) ^ TurnEast(t)) => (FacingEast(t+1) ^ L(x,y,t+1))
                    clauses.append([-L(x, y, t), -TurnEast(t), FacingEast(t + 1)])
                    clauses.append([-L(x, y, t), -TurnEast(t), L(x, y, t + 1)])

                    # (L(x,y,t) ^ TurnWest(t)) => (FacingWest(t+1) ^ L(x,y,t+1))
                    clauses.append([-L(x, y, t), -TurnWest(t), FacingWest(t + 1)])
                    clauses.append([-L(x, y, t), -TurnWest(t), L(x, y, t + 1)])

                    # (L(x,y,t) ^ TurnNorth(t)) => (FacingNorth(t+1) ^ L(x,y,t+1))
                    clauses.append([-L(x, y, t), -TurnNorth(t), FacingNorth(t + 1)])
                    clauses.append([-L(x, y, t), -TurnNorth(t), L(x, y, t + 1)])

                    # (L(x,y,t) ^ TurnSouth(t)) => (FacingSouth(t+1) ^ L(x,y,t+1))
                    clauses.append([-L(x, y, t), -TurnSouth(t), FacingSouth(t + 1)])
                    clauses.append([-L(x, y, t), -TurnSouth(t), L(x, y, t + 1)])

                    # (x,y)!=(x1,y1) L(x,y,t) ^ L(x1,y1,t) => Forward(t)
                    for x2 in range(n):
                        for y2 in range(n):
                            if (x, y) != (x2, y2):
                                clauses.append([-L(x, y, t), -L(x2, y2, t + 1), Forward(t)])

        # Append the current face direction to the clauses list.
        clauses.append([current_face_])
        # Append the starting location of the agent at time step 0 to the clauses list.
        clauses.append([L(current_loc[0], current_loc[1], 0)])
        # goals that no longer exist
        # clauses.append([L(des_loc[0], des_loc[1], T - 1) for des_loc in goals_locs])

        # Initialize an empty list to hold the plan of actions.
        plan = []
        cnf = CNF(from_clauses=clauses)  # knowledge base
        with Solver(name='Cadical195', bootstrap_with=cnf) as solver:
            if solver.solve():
                for m in solver.get_model():
                    if m > 0:
                        item = vpool.id2obj[m].split('@')
                        # print(item)
                        if item[0] in ['Forward', 'TurnWest', 'TurnEast', 'TurnNorth', 'TurnSouth']:
                            plan.append((item[0], int(item[1])))

                # Sort the plan by the time step to execute actions in the correct order.
                plan.sort(key=lambda x: x[1])

        # Check if exactly one action is planned.
        if len(plan) == 1:
            # If the planned action is a turn and the agent is already facing the correct direction, return 'Shoot'.
            if (plan[0][0] == 'TurnWest' and current_face == "FacingWest") or \
                    (plan[0][0] == 'TurnEast' and current_face == "FacingEast") or \
                    (plan[0][0] == 'TurnNorth' and current_face == "FacingNorth") or \
                    (plan[0][0] == 'TurnSouth' and current_face == "FacingSouth") or \
                    (plan[0][0] == 'Forward'):
                return [('Shoot', 0)] # Return a shooting action with time step 0.

        # If no plan is found, return None; otherwise, append 'Shoot' action to the end of the plan.
        return None if len(plan) == 0 else plan + [('Shoot', plan[-1][1] + 1)]



    optimal_plan = None
    for t in range(1, T + 1):
        plan = PlanShot_(current_face, current_loc, goals_locs, safe_squares, n, t)
        if plan is not None:
            if optimal_plan is None:
                optimal_plan = plan
                break
    return optimal_plan

def PlanRoute(current_face,current_loc,goals_locs,safe_squares,n=4,T=100):
    def PlanRoute_(current_face, current_loc, goals_locs, safe_squares, n=4, T=10):
        """
        Plan the route for an agent to reach a goal within a grid.

        :param current_face: one from ['FacingWest', 'FacingNorth','FacingEast','FacingSouth']
        :param current_loc: (x,y) current agent's location
        :param goals_locs: (x2,y2) location to reach
        :param safe_squares: {(w,e),(r,t),...} squares known to be safe
        :param n: grid nxn
        :param T: time steps (exact number of actions) to reach the goal
        :return: A plan with the sequence of actions or None if no solution
        """

        # Create a unique ID pool starting from 1 to manage variable assignments
        vpool = IDPool(start_from=1)

        # Define all squares in the grid as (x, y) tuples
        all_squares = set([(i, j) for i in range(n) for j in range(n)])

        # Define the variable for forward movement action
        Forward = lambda t: vpool.id(f'Forward@{t}')

        # Define turn actions (East, West, North, South) at time step t
        TurnEast = lambda t: vpool.id(f'TurnEast@{t}')
        TurnWest = lambda t: vpool.id(f'TurnWest@{t}')
        TurnNorth = lambda t: vpool.id(f'TurnNorth@{t}')
        TurnSouth = lambda t: vpool.id(f'TurnSouth@{t}')

        # Define location state at time t (whether agent is at location x, y at time t)
        L = lambda x, y, t: vpool.id(f'L@{x}@{y}@{t}')

        # Define facing directions at time t (West, North, East, South)
        FacingWest = lambda t: vpool.id(f'FacingWest@{t}')
        FacingNorth = lambda t: vpool.id(f'FacingNorth@{t}')
        FacingEast = lambda t: vpool.id(f'FacingEast@{t}')
        FacingSouth = lambda t: vpool.id(f'FacingSouth@{t}')

        # Loop over all time steps to initialize all variables
        for t in range(T):
            Forward(t)
            TurnEast(t)
            TurnWest(t)
            TurnNorth(t)
            TurnSouth(t)

            FacingWest(t)
            FacingNorth(t)
            FacingEast(t)
            FacingSouth(t)

            for x in range(n):
                for y in range(n):
                    for t in range(T):
                        L(x, y, t)

        clauses = []

        # Set the initial facing direction based on current_face
        if current_face == "FacingEast":
            current_face_ = FacingEast(0)
        elif current_face == "FacingWest":
            current_face_ = FacingWest(0)
        elif current_face == "FacingNorth":
            current_face_ = FacingNorth(0)
        elif current_face == "FacingSouth":
            current_face_ = FacingSouth(0)

        # Mark all not-safe squares, the agent should avoid these squares
        not_safe = all_squares.difference(safe_squares)
        for ns in not_safe:
            for t in range(T - 1):
                clauses.append([-L(ns[0], ns[1], t)])

        # Ensure exactly one facing direction at each time step (the agent can only face one direction)
        Orientations = [FacingWest, FacingNorth, FacingEast, FacingSouth]
        for t in range(T):
            clauses.append([o(t) for o in Orientations])
            for o1 in range(len(Orientations)):
                for o2 in range(o1 + 1, len(Orientations)):
                    clauses.append([-Orientations[o1](t), -Orientations[o2](t)])

        # Ensure exactly one location is occupied by the agent at each time step
        for t in range(T):
            clauses.append([L(x, y, t) for x in range(n) for y in range(n)])
            for x1 in range(n):
                for y1 in range(n):
                    for x2 in range(n):
                        for y2 in range(n):
                            if (x1, y1) != (x2, y2):
                                clauses.append([-L(x1, y1, t), -L(x2, y2, t)])

        # Ensure exactly one action is performed at each time step
        Actions = [Forward, TurnEast, TurnWest, TurnNorth, TurnSouth]
        for t in range(T - 1):
            clauses.append([o(t) for o in Actions])
            for o1 in range(len(Actions)):
                for o2 in range(o1 + 1, len(Actions)):
                    clauses.append([-Actions[o1](t), -Actions[o2](t)])

        # At the final time step, ensure no actions are performed.
        # In other words, the agent must have reached its goal by this time step,
        # or it has failed to reach the goal within the specified time frame.
        for action in Actions:
            clauses.append([-action(T - 1)])

        # Define the consequences of each action
        for t in range(T - 1):

            # If an action is taken, ensure the agent faces the correct direction in the next time step

            clauses.append([-FacingWest(t), TurnEast(t), TurnNorth(t), TurnSouth(t), FacingWest(t + 1)])
            clauses.append([-FacingEast(t), TurnWest(t), TurnNorth(t), TurnSouth(t), FacingEast(t + 1)])
            clauses.append([-FacingNorth(t), TurnEast(t), TurnWest(t), TurnSouth(t), FacingNorth(t + 1)])
            clauses.append([-FacingSouth(t), TurnEast(t), TurnWest(t), TurnNorth(t), FacingSouth(t + 1)])

            clauses.append([-FacingWest(t + 1), TurnWest(t), FacingWest(t)])
            clauses.append([-FacingEast(t + 1), TurnEast(t), FacingEast(t)])
            clauses.append([-FacingNorth(t + 1), TurnNorth(t), FacingNorth(t)])
            clauses.append([-FacingSouth(t + 1), TurnSouth(t), FacingSouth(t)])

            # Consequences of forward movement (moving to neighboring squares based on current direction)
            for x in range(n):
                for y in range(n):
                    # Forward

                    # FacingWest
                    if y - 1 >= 0:
                        # A ∧ B ∧ C → (D ∧ ¬A ) = (D ∨ ¬A ∨ ¬B ∨ ¬C) ∧ (¬A ∨ ¬B ∨ ¬C)
                        clauses.append([L(x, y - 1, t + 1), -L(x, y, t), -FacingWest(t), -Forward(t)])
                    else:
                        clauses.append([-L(x, y, t), -FacingWest(t), -Forward(t), L(x, y, t + 1)])
                        clauses.append([-L(x, y, t), -FacingWest(t), -Forward(t)])

                    # FacingEast
                    if y + 1 < n:
                        clauses.append([L(x, y + 1, t + 1), -L(x, y, t), -FacingEast(t), -Forward(t)])
                    else:
                        clauses.append([-L(x, y, t), -FacingEast(t), -Forward(t), L(x, y, t + 1)])
                        clauses.append([-L(x, y, t), -FacingEast(t), -Forward(t)])

                    # FacingNorth
                    if x - 1 >= 0:
                        clauses.append([L(x - 1, y, t + 1), -L(x, y, t), -FacingNorth(t), -Forward(t)])
                    else:
                        clauses.append([-L(x, y, t), -FacingNorth(t), -Forward(t), L(x, y, t + 1)])
                        clauses.append([-L(x, y, t), -FacingNorth(t), -Forward(t)])

                    # FacingSouth
                    if x + 1 < n:
                        clauses.append([L(x + 1, y, t + 1), -L(x, y, t), -FacingSouth(t), -Forward(t)])
                    else:
                        clauses.append([-L(x, y, t), -FacingSouth(t), -Forward(t), L(x, y, t + 1)])
                        clauses.append([-L(x, y, t), -FacingSouth(t), -Forward(t)])

                    # Actions resulting in change of orientation and location (for each turn action)

                    # (L(x,y,t) ^ TurnEast(t)) => (FacingEast(t+1) ^ L(x,y,t+1))
                    clauses.append([-L(x, y, t), -TurnEast(t), FacingEast(t + 1)])
                    clauses.append([-L(x, y, t), -TurnEast(t), L(x, y, t + 1)])

                    # (L(x,y,t) ^ TurnWest(t)) => (FacingWest(t+1) ^ L(x,y,t+1))
                    clauses.append([-L(x, y, t), -TurnWest(t), FacingWest(t + 1)])
                    clauses.append([-L(x, y, t), -TurnWest(t), L(x, y, t + 1)])

                    # (L(x,y,t) ^ TurnNorth(t)) => (FacingNorth(t+1) ^ L(x,y,t+1))
                    clauses.append([-L(x, y, t), -TurnNorth(t), FacingNorth(t + 1)])
                    clauses.append([-L(x, y, t), -TurnNorth(t), L(x, y, t + 1)])

                    # (L(x,y,t) ^ TurnSouth(t)) => (FacingSouth(t+1) ^ L(x,y,t+1))
                    clauses.append([-L(x, y, t), -TurnSouth(t), FacingSouth(t + 1)])
                    clauses.append([-L(x, y, t), -TurnSouth(t), L(x, y, t + 1)])

                    # (x,y)!=(x1,y1) L(x,y,t) ^ L(x1,y1,t) => Forward(t)
                    for x2 in range(n):
                        for y2 in range(n):
                            if (x, y) != (x2, y2):
                                clauses.append([-L(x, y, t), -L(x2, y2, t + 1), Forward(t)])

        # Set initial state: starting position, facing direction, and goals
        clauses.append([current_face_])
        clauses.append([L(current_loc[0], current_loc[1], 0)])
        clauses.append([L(des_loc[0], des_loc[1], T - 1) for des_loc in goals_locs])

        # Initialize the solver with the clauses (knowledge base) and solve the problem using a SAT solver
        plan = []
        cnf = CNF(from_clauses=clauses)
        with Solver(name='Cadical195', bootstrap_with=cnf) as solver:
            if solver.solve():
                for m in solver.get_model():
                    if m > 0:
                        item = vpool.id2obj[m].split('@')
                        if item[0] in ['Forward', 'TurnWest', 'TurnEast', 'TurnNorth', 'TurnSouth']:
                            plan.append((item[0], int(item[1])))

                plan.sort(key=lambda x: x[1])

        # An action plan that only changes the direction without altering the agent's location.
        # Assuming the current location is not a goal state, we can ignore this plan.
        if len(plan) == 1:
            if plan[0][0] in ['TurnWest', 'TurnEast', 'TurnNorth', 'TurnSouth']:
                return []

        return None if len(plan) == 0 else plan

    optimal_plan = None
    for t in range(1, T + 1):
        plan = PlanRoute_(current_face, current_loc, goals_locs, safe_squares, n, t)
        if plan is not None:
            if optimal_plan is None:
                optimal_plan = plan
                break
    return optimal_plan


class Agent:
    def __init__(self,grid_size=4,max_steps=50):
        self.agent_face="FacingEast"
        self.grid_size=grid_size
        self.max_steps=max_steps
        self.query_database=WumpusWorldQueryDatabase(self.grid_size)
        self.plan=[]

    def next_action(self,percept):
        if self.plan is None or len(self.plan)==0:
            plan = None
            safe = self.query_database.get_safe_cells()

            if percept['Glitter']:
                plan = PlanRoute(self.agent_face, percept['AgentLoc'], {(0, 0)}, safe,self.grid_size,self.max_steps)
                if plan is not None:
                    plan = ['Grab'] + [x for x, y in plan] + ['Climb']
            if plan is None:
                unvisited = self.query_database.get_unvisited_cells()
                glocs=safe.intersection(unvisited)
                if len(glocs)>0:
                    plan = PlanRoute(self.agent_face, percept['AgentLoc'], safe.intersection(unvisited), safe,self.grid_size,self.max_steps)
                    if plan is not None:
                        plan = [x for x, y in plan]

            if plan is None and self.query_database.has_arrow:
                possible_wumpus = self.query_database.get_possible_wumpus_locations()
                plan = PlanShot(self.agent_face, percept['AgentLoc'], possible_wumpus, safe,self.grid_size,self.max_steps)
                if plan is not None:
                    plan = [x for x, y in plan]

            if plan is None:
                plan = PlanRoute(self.agent_face, percept['AgentLoc'], {(0, 0)}, safe,self.grid_size,self.max_steps)
                plan = [x for x, y in plan] + ['Climb']

            self.plan = plan


        action=self.plan.pop(0)
        if action.startswith("Turn"):
            self.agent_face = f"Facing{action[4:]}"
        return action


# Initialize pygame
pygame.init()

# Initialize the mixer module
pygame.mixer.init()

# Load sounds
move_sound = pygame.mixer.Sound('8-bit-move-001-171817.mp3')
gold_sound = pygame.mixer.Sound('bonus-points-190035.mp3')
wall_sound = pygame.mixer.Sound('wall-hit-3-48114.mp3')
arrow_impact_sound = pygame.mixer.Sound('arrow-impact-87260.mp3')
game_character_scream = pygame.mixer.Sound('game-character-scream-131144.mp3')
game_music_loop = pygame.mixer.Sound('game-music-loop-6-144641.mp3')
game_start_sound = pygame.mixer.Sound('game-start-317318.mp3')

# You may want to adjust the volume
move_sound.set_volume(0.1)
gold_sound.set_volume(0.5)
wall_sound.set_volume(1)
arrow_impact_sound.set_volume(0.5)
game_character_scream.set_volume(0.5)
game_music_loop.set_volume(0.4)
game_start_sound.set_volume(0.5)

visited_cells={(0,0)}

class WumpusWorldSimulator:
    def __init__(self, n=4, Gold=(2, 1), Wampus=(2, 0), Pit={(0, 2), (2, 2), (3, 3)}):
        self.n = n  # Grid size (n x n)
        self.Wampus = Wampus  # Wumpus location
        self.Pit = Pit  # Set of pit locations
        self.WampusAlive = True  # Status of Wumpus
        self.Gold = Gold  # Gold location
        self.AgentHasArrow = True  # Whether the agent still has an arrow
        self.AgentLoc = (0, 0)  # Agent's starting location
        self.AgentFace = "FacingEast"  # Agent's initial direction
        self.Finished = False  # Game status
        self.score = 0  # Agent's score

        # Initialize pit grid
        self.Pit = [[False for _ in range(self.n)] for _ in range(self.n)]
        for p in Pit:
            self.Pit[p[0]][p[1]] = True

        # Initialize breeze grid (adjacent to pits)
        self.Breeze = [[False for _ in range(self.n)] for _ in range(self.n)]
        for p in Pit:
            for n in self.get_neighbors(p[0], p[1]):
                self.Breeze[n[0]][n[1]] = True

        # Initialize stench grid (adjacent to Wumpus)
        self.Stench = [[False for _ in range(self.n)] for _ in range(self.n)]
        for n in self.get_neighbors(Wampus[0], Wampus[1]):
            self.Stench[n[0]][n[1]] = True

        # Initial agent perception
        self.recent_percept = {
            'Glitter': self.AgentLoc == self.Gold,  # Detect gold
            'Stench': self.Stench[self.AgentLoc[0]][self.AgentLoc[1]],  # Detect Wumpus nearby
            'Breeze': self.Breeze[self.AgentLoc[0]][self.AgentLoc[1]],  # Detect pit nearby
            'Scream': False,  # Wumpus dying sound
            'Bump': False,  # Collision with a wall
            'AgentLoc': self.AgentLoc,  # Current agent position
            'Bang': False  # Arrow shot status
        }

    def get_neighbors(self, x, y):
        return [(x + dx, y + dy) for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                if 0 <= x + dx < self.n and 0 <= y + dy < self.n]

    def update(self, action):
        if action.startswith("Turn"):
            self.AgentFace = f"Facing{action[4:]}"
            move_sound.play()

        elif action == "Forward":
            if self.AgentFace == "FacingEast" and (self.AgentLoc[1] + 1 < self.n):
                self.AgentLoc = (self.AgentLoc[0], self.AgentLoc[1] + 1)
                move_sound.play()
            elif self.AgentFace == "FacingWest" and (self.AgentLoc[1] - 1 >= 0):
                self.AgentLoc = (self.AgentLoc[0], self.AgentLoc[1] - 1)
                move_sound.play()
            elif self.AgentFace == "FacingSouth" and (self.AgentLoc[0] + 1 < self.n):
                self.AgentLoc = (self.AgentLoc[0] + 1, self.AgentLoc[1])
                move_sound.play()
            elif self.AgentFace == "FacingNorth" and (self.AgentLoc[0] - 1 >= 0):
                self.AgentLoc = (self.AgentLoc[0] - 1, self.AgentLoc[1])
                move_sound.play()
            else:
                self.recent_percept['Bump'] = True
                wall_sound.play()
        elif action == "Grab" and self.AgentLoc == self.Gold:
            self.Gold = (-10, -10)
            self.recent_percept['Glitter'] = False
            gold_sound.play()
        elif action == "Shoot" and self.AgentHasArrow:
            self.score -= 10
            self.AgentHasArrow = False
            if self.WampusAlive:
                if (self.AgentFace == "FacingEast" and self.AgentLoc[0] == self.Wampus[0] and self.AgentLoc[1] < self.Wampus[1]) or \
                   (self.AgentFace == "FacingWest" and self.AgentLoc[0] == self.Wampus[0] and self.AgentLoc[1] > self.Wampus[1]) or \
                   (self.AgentFace == "FacingSouth" and self.AgentLoc[1] == self.Wampus[1] and self.AgentLoc[0] < self.Wampus[0]) or \
                   (self.AgentFace == "FacingNorth" and self.AgentLoc[1] == self.Wampus[1] and self.AgentLoc[0] > self.Wampus[0]):
                    self.WampusAlive = False
                    self.recent_percept['Scream'] = True
                    game_character_scream.play()
                else:
                    arrow_impact_sound.play()
            else:
                arrow_impact_sound.play()
        elif action == "Climb" and self.AgentLoc == (0, 0):
            self.Finished = True
            if self.Gold == (-10, -10):
                self.score += 1000
            game_start_sound.play()

        self.recent_percept['Bang'] = action == "Shoot"
        self.recent_percept['Breeze'] = self.Breeze[self.AgentLoc[0]][self.AgentLoc[1]]
        self.recent_percept['Stench'] = self.Stench[self.AgentLoc[0]][self.AgentLoc[1]]
        self.recent_percept['AgentLoc'] = self.AgentLoc
        self.recent_percept['Glitter'] = self.AgentLoc == self.Gold

        if (self.AgentLoc == self.Wampus and self.WampusAlive) or (self.Pit[self.AgentLoc[0]][self.AgentLoc[1]]):
            self.Finished = True
            self.score -= 1000
            game_start_sound.play()

        print(action,self.AgentFace,self.AgentLoc)

        self.score -= 1
        return self.recent_percept





# Set up the screen
screen = pygame.display.set_mode((400, 400))
pygame.display.set_caption("Wumpus World Simulator")

# Set up fonts
font = pygame.font.Font(None, 55)

cell_green_img = pygame.image.load('CellGreen.png')
cell_green_img = pygame.transform.scale(cell_green_img, (100, 100))

stench_img = pygame.image.load('Stench.png')
stench_img = pygame.transform.scale(stench_img, (100, 100))

breeze_img = pygame.image.load('Breeze.png')
breeze_img = pygame.transform.scale(breeze_img, (100, 100))

pit_img = pygame.image.load('Pit.png')
pit_img = pygame.transform.scale(pit_img, (100, 100))

gold_img = pygame.image.load('Gold.png')
gold_img = pygame.transform.scale(gold_img, (100, 100))

# Load Wumpus images (alive and dead)
wumpus_alive_img = pygame.image.load('WampusAlive.png')
wumpus_alive_img = pygame.transform.scale(wumpus_alive_img, (100, 100))

wumpus_dead_img = pygame.image.load('WampusDead.png')
wumpus_dead_img = pygame.transform.scale(wumpus_dead_img, (100, 100))


# Load Agent Images (based on direction and items)
agent_images = {
    "FacingWest": {
        "default": pygame.image.load('AgentWest.png'),
        "with_arrow": pygame.image.load('AgentWestWithArrow.png'),
        "with_gold": pygame.image.load('AgentWestWithGold.png'),
        "with_both": pygame.image.load('AgentWestWithArrowAndGold.png')
    },
    "FacingEast": {
        "default": pygame.image.load('AgentEast.png'),
        "with_arrow": pygame.image.load('AgentEastWithArrow.png'),
        "with_gold": pygame.image.load('AgentEastWithGold.png'),
        "with_both": pygame.image.load('AgentEastWithArrowAndGold.png')
    },
    "FacingSouth": {
        "default": pygame.image.load('AgentSouth.png'),
        "with_arrow": pygame.image.load('AgentSouthWithArrow.png'),
        "with_gold": pygame.image.load('AgentSouthWithGold.png'),
        "with_both": pygame.image.load('AgentSouthWithArrowAndGold.png')
    },
    "FacingNorth": {
        "default": pygame.image.load('AgentNorth.png'),
        "with_arrow": pygame.image.load('AgentNorthWithArrow.png'),
        "with_gold": pygame.image.load('AgentNorthWithGold.png'),
        "with_both": pygame.image.load('AgentNorthWithArrowAndGold.png')
    }
}
for direction in agent_images:
    for state in agent_images[direction]:
        agent_images[direction][state] = pygame.transform.scale(agent_images[direction][state], (100, 100))

def get_agent_image(agent):
    direction = agent.AgentFace
    if agent.Gold == (-10, -10) and agent.AgentHasArrow:
        return agent_images[direction]["with_both"]
    elif agent.Gold == (-10, -10):
        return agent_images[direction]["with_gold"]
    elif agent.AgentHasArrow:
        return agent_images[direction]["with_arrow"]
    return agent_images[direction]["default"]

# Initialize the Wumpus World
agent = Agent()
game = WumpusWorldSimulator()
percept = game.recent_percept
agent.query_database.update_knowledge_base_with_percept(percept)


game_start_sound.play()
game_music_loop.play(loops=-1)
running = True
# Game loop
while running:
    screen.fill((255, 255, 255))

    if manual_mode and game.Finished==False:
        # Manual control: user presses keys
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    game.update("Shoot")
                elif event.key == pygame.K_ESCAPE:
                    game.update("Climb")
                elif event.key == pygame.K_g:
                    game.update("Grab")
                elif event.key == pygame.K_w:
                    game.update("TurnNorth")
                elif event.key == pygame.K_d:
                    game.update("TurnEast")
                elif event.key == pygame.K_s:
                    game.update("TurnSouth")
                elif event.key == pygame.K_a:
                    game.update("TurnWest")
                elif event.key == pygame.K_SPACE:
                    game.update("Forward")

    elif game.Finished==False:
        # Agent mode: agent takes actions automatically
        action = agent.next_action(percept)
        percept = game.update(action)
        agent.query_database.update_knowledge_base_with_percept(percept)

    visited_cells.add(game.AgentLoc)

    # Draw the grid and entities
    for row in range(game.n):
        for col in range(game.n):
            screen.blit(cell_green_img, (col * 100, row * 100))

            if (row, col) in visited_cells:
                pygame.draw.rect(screen, (0, 0, 0), (col * 100, row * 100, 100, 100), 1)  # Black border

            if game.Stench[row][col]:
                screen.blit(stench_img, (col * 100, row * 100))

            if game.Breeze[row][col]:
                screen.blit(breeze_img, (col * 100, row * 100))

            if game.Pit[row][col]:
                screen.blit(pit_img, (col * 100, row * 100))

            if (row, col) == game.Gold:
                screen.blit(gold_img, (col * 100, row * 100))

            if (row, col) == game.Wampus:
                if game.WampusAlive:
                    screen.blit(wumpus_alive_img, (col * 100, row * 100))
                else:
                    screen.blit(wumpus_dead_img, (col * 100, row * 100))

    # Draw the agent
    agent_img = get_agent_image(game)
    screen.blit(agent_img, (game.AgentLoc[1] * 100, game.AgentLoc[0] * 100))

    # Display "Game Over" and score if finished
    if game.Finished:
        # Create a semi-transparent overlay
        overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)  # Enable alpha
        overlay.fill((243, 231, 231, 150))

        # Draw the overlay on the screen
        screen.blit(overlay, (0, 0))
        game_over_text = font.render("Game Over", True, (255, 68, 51))
        screen.blit(game_over_text, (120, 150))

        score_text = font.render(f"Score: {game.score}", True, (255, 68, 51))
        screen.blit(score_text, (140, 200))

    pygame.display.flip()
    if manual_mode==False:
        sleep_for_agent_mode()

pygame.quit()
sys.exit()

