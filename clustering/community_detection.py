import community as community_louvain
from collections import defaultdict
import networkx as nx


def louvain_method(G):
    partition = community_louvain.best_partition(G)

    clusters = defaultdict(list)
    for word, cluster_id in partition.items():
        clusters[cluster_id].append(word)

    return clusters, partition


def girvan_newman_method(G, level=1):
    comp = nx.community.girvan_newman(G)

    for _ in range(level):
        communities = next(comp)

    clusters = defaultdict(list)
    partition = {}

    for i, community in enumerate(communities):
        for node in community:
            clusters[i].append(node)
            partition[node] = i

    return clusters, partition


def detect_communities(G, method="louvain"):
    if method == "louvain":
        return louvain_method(G)
    elif method == "girvan_newman":
        return girvan_newman_method(G)
    else:
        raise ValueError("Unknown method")