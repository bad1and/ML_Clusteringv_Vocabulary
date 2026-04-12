import matplotlib.pyplot as plt
import networkx as nx

def draw_graph(G, partition):
    pos = nx.spring_layout(G)
    colors = [partition[node] for node in G.nodes()]

    nx.draw(G, pos, with_labels=True, node_color=colors)
    plt.show()