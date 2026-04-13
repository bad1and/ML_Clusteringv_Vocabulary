import networkx as nx
from config import WINDOW_SIZE

def build_graph(words):
    G = nx.Graph()

    for i in range(len(words)):
        for j in range(i + 1, i + WINDOW_SIZE):
            if j < len(words):
                w1, w2 = words[i], words[j]

                distance = j - i
                weight = 1 / distance  # усиление близких слов

                if G.has_edge(w1, w2):
                    G[w1][w2]["weight"] += weight
                else:
                    G.add_edge(w1, w2, weight=weight)

    return G