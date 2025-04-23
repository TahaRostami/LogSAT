from pysat.formula import CNF
from pysat.formula import IDPool

nationality=['Brit','Swede','Dane','Norwegian','German']
colour=['red','green','white','yellow','blue']
drinks=['tea','coffee','milk','beer','water']
cigarette=['Pall Mall','Dunhill','Blends','Blue Master','Prince']
pet=['dogs','birds','cats','horses','fish']

A=nationality+colour+drinks+cigarette+pet

vpool = IDPool(start_from=1)
S = lambda i, j: vpool.id('S{0}@{1}'.format(i, j))

for i in range(1,6):
    for k in A:
        S(i,k)
clauses=[]

# The knowledge that each attribute appears at
# least once can be encoded as the clauses
for a in A:
    clauses.append([vpool.id(f'S{i}@{a}') for i in range(1,6)])


#the knowledge that each attribute is not shared can be encoded
for a in A:
    for i in range(1, 6):
        for j in range(1, 6):
            if i != j:
                clauses.append([-vpool.id(f'S{i}@{a}'), -vpool.id(f'S{j}@{a}')])



#the fact that each house has some colour is encoded as
for i in range(1,6):
    clauses.append([vpool.id(f'S{i}@{a}') for a in colour])
# the knowledge that each house cannot have two colours can be encoded
for i in range(1, 6):
    for c in colour:
        for d in colour:
            if c != d:
                clauses.append([-vpool.id(f'S{i}@{c}'), -vpool.id(f'S{i}@{d}')])



#the fact that each house has some nationality is encoded as
for i in range(1,6):
    clauses.append([vpool.id(f'S{i}@{a}') for a in nationality])
# the knowledge that each house cannot have two nationality can be encoded
for i in range(1, 6):
    for c in nationality:
        for d in nationality:
            if c != d:
                clauses.append([-vpool.id(f'S{i}@{c}'), -vpool.id(f'S{i}@{d}')])


#the fact that each house has some drinks is encoded as
for i in range(1,6):
    clauses.append([vpool.id(f'S{i}@{a}') for a in drinks])
# the knowledge that each house cannot have two drinks can be encoded
for i in range(1, 6):
    for c in drinks:
        for d in drinks:
            if c != d:
                clauses.append([-vpool.id(f'S{i}@{c}'), -vpool.id(f'S{i}@{d}')])


#the fact that each house has some cigarette is encoded as
for i in range(1,6):
    clauses.append([vpool.id(f'S{i}@{a}') for a in cigarette])
# the knowledge that each house cannot have two cigarette can be encoded
for i in range(1, 6):
    for c in cigarette:
        for d in cigarette:
            if c != d:
                clauses.append([-vpool.id(f'S{i}@{c}'), -vpool.id(f'S{i}@{d}')])


#the fact that each house has some pet is encoded as
for i in range(1,6):
    clauses.append([vpool.id(f'S{i}@{a}') for a in pet])
# the knowledge that each house cannot have two cigarette can be encoded
for i in range(1, 6):
    for c in pet:
        for d in pet:
            if c != d:
                clauses.append([-vpool.id(f'S{i}@{c}'), -vpool.id(f'S{i}@{d}')])


#1. The Brit lives in the red house.
for i in range(1,6):
    clauses.append([-vpool.id(f'S{i}@Brit'),vpool.id(f'S{i}@red')])

#2. The Swede keeps dogs as pets.
for i in range(1,6):
    clauses.append([-vpool.id(f'S{i}@Swede'),vpool.id(f'S{i}@dogs')])

#3. The Dane drinks tea.
for i in range(1,6):
    clauses.append([-vpool.id(f'S{i}@Dane'),vpool.id(f'S{i}@tea')])

#4. The green house is next to the white house, on the left.
for i in range(1,5):
    clauses.append([-vpool.id(f'S{i}@green'),vpool.id(f'S{i+1}@white')])
clauses.append([-vpool.id(f'S5@green')])

#5. The owner of the green house drinks coffee.
for i in range(1,6):
    clauses.append([-vpool.id(f'S{i}@green'),vpool.id(f'S{i}@coffee')])

#6. The person who smokes Pall Mall rears birds.
for i in range(1,6):
    clauses.append([-vpool.id(f'S{i}@Pall Mall'),vpool.id(f'S{i}@birds')])

#7. The owner of the yellow house smokes Dunhill.
for i in range(1,6):
    clauses.append([-vpool.id(f'S{i}@yellow'),vpool.id(f'S{i}@Dunhill')])

#8. The man living in the centre house drinks milk.
clauses.append([vpool.id('S3@milk')])

#9. The Norwegian lives in the first house
clauses.append([vpool.id('S1@Norwegian')])

#10. The man who smokes Blends lives next to the one who keeps cats.
#next to == left or right close neighborhood
for i in range(2,5):
    clauses.append([-vpool.id(f'S{i}@Blends'),vpool.id(f'S{i-1}@cats'),vpool.id(f'S{i+1}@cats')])

clauses.append([-vpool.id(f'S1@Blends'),vpool.id(f'S2@cats')])
clauses.append([-vpool.id(f'S5@Blends'),vpool.id(f'S4@cats')])

#11. The man who keeps horses lives next to the man who smokes Dunhill.
for i in range(2,5):
    clauses.append([-vpool.id(f'S{i}@horses'),vpool.id(f'S{i-1}@Dunhill'),vpool.id(f'S{i+1}@Dunhill')])

clauses.append([-vpool.id(f'S1@horses'),vpool.id(f'S2@Dunhill')])
clauses.append([-vpool.id(f'S5@horses'),vpool.id(f'S4@Dunhill')])

#12. The man who smokes Blue Master drinks beer.
for i in range(1,6):
    clauses.append([-vpool.id(f'S{i}@Blue Master'),vpool.id(f'S{i}@beer')])

#13. The German smokes Prince.
for i in range(1,6):
    clauses.append([-vpool.id(f'S{i}@German'),vpool.id(f'S{i}@Prince')])

#14. The Norwegian lives next to the blue house.
for i in range(2,5):
    clauses.append([-vpool.id(f'S{i}@Norwegian'),vpool.id(f'S{i-1}@blue'),vpool.id(f'S{i+1}@blue')])

clauses.append([-vpool.id(f'S1@Norwegian'),vpool.id(f'S2@blue')])
clauses.append([-vpool.id(f'S5@Norwegian'),vpool.id(f'S4@blue')])

#15. The man who smokes Blends has a neighbour who drinks water.
for i in range(2,5):
    clauses.append([-vpool.id(f'S{i}@Blends'),vpool.id(f'S{i-1}@water'),vpool.id(f'S{i+1}@water')])

clauses.append([-vpool.id(f'S1@Blends'),vpool.id(f'S2@water')])
clauses.append([-vpool.id(f'S5@Blends'),vpool.id(f'S4@water')])


from pysat.solvers import Solver
cnf = CNF(from_clauses=clauses)
models = []
with Solver(bootstrap_with=cnf) as solver:
    while solver.solve():
        model = solver.get_model()
        models.append(model)

        print('-' * 50)
        for i in range(1, 6):
            print(f'House {i}:')
            for category in [nationality, colour, drinks, cigarette, pet]:
                for item in category:
                    if vpool.id(f'S{i}@{item}') in model:
                        print(f'  {item}')
            print('-' * 50)

        # Add a blocking clause to prevent this model from appearing again
        solver.add_clause([-lit for lit in model])

print(f'Total models found: {len(models)}')
