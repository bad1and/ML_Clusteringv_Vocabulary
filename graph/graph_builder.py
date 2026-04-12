import networkx as nx

def build_graph(words, window_size=3):
    G = nx.Graph()

    for i in range(len(words)):
        for j in range(i + 1, i + window_size):
            if j < len(words):
                w1, w2 = words[i], words[j]

                if G.has_edge(w1, w2):
                    G[w1][w2]["weight"] += 1
                else:
                    G.add_edge(w1, w2, weight=1)

    return G