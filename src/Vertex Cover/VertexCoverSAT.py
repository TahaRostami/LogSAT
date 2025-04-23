import matplotlib.pyplot as plt
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
from pysat.formula import IDPool
import networkx as nx
import numpy as np

G=np.array([[0,1,0,0,0,1],
            [1,0,1,1,1,1],
            [0,1,0,0,0,0],
            [0,1,0,0,0,0],
            [0,1,0,0,0,0],
            [1,1,0,0,0,0]])



G = nx.from_numpy_array(G)


vpool = IDPool(start_from=1)
# Whether the i-th vertex is selected
V=lambda i:vpool.id(f"V@{i}")

wcnf=WCNF()

# Ensure at least one endpoint of each edge is in the vertex cover
for u,v in G.edges:
    wcnf.append([V(u),V(v)])

# Soft constraints to minimize the number of selected vertices
for i in G.nodes:
    wcnf.append([-V(i)],weight=1)

vertices=None
with RC2(wcnf) as rc2:
    model = rc2.compute()
    total_W,total_P = 0,0
    if model is not None:
        print("SAT")
        vertices=[int(vpool.id2obj[m].split('@')[1]) for m in model if m > 0 and m in vpool.id2obj]
        print(len(vertices))
    else:
        print("UNSAT")

if vertices is not None:
    plt.figure(figsize=(6, 6))

    node_colors = ['lightgreen' if node in vertices else 'gray' for node in G.nodes]
    nx.draw(G, with_labels=True, node_color=node_colors, edge_color='yellow',
            node_size=700, font_size=12)

    plt.show()