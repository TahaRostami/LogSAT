import matplotlib.pyplot as plt
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
from pysat.formula import IDPool
import rustworkx.generators
from rustworkx.visualization import mpl_draw

G = rustworkx.generators.generalized_petersen_graph(12, 4)

vpool = IDPool(start_from=1)
# Whether the i-th vertex is selected
V=lambda i:vpool.id(f"V@{i}")

wcnf=WCNF()

# If an edge exists in the graph, both vertices cannot be chosen
for u,v in G.edge_list():
    wcnf.append([-V(u),-V(v)])

# Soft constraints to maximize the independent set's size
for i in G.node_indices():
    wcnf.append([V(i)],weight=1)

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

    node_colors = ['lightgreen' if node in vertices else 'gray' for node in G.node_indices()]
    layout = rustworkx.shell_layout(
      G, nlist=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [15, 16, 17, 18, 19,20,21,22,23,12, 13, 14]]
    )

    mpl_draw(G, pos=layout,node_color=node_colors,edge_color='yellow',with_labels=True)

    plt.show()