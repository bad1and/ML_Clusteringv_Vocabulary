import matplotlib.pyplot as plt
import networkx as nx

def draw_graph(G, partition, title="Graph"):
    pos = nx.spring_layout(G)

    colors = [partition[node] for node in G.nodes()]
    weights = [G[u][v]["weight"] for u, v in G.edges()]

    plt.figure(figsize=(10, 7))
    plt.title(title)

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=colors,
        node_size=500,
        font_size=8,
        width=weights
    )

    plt.show(block=False)