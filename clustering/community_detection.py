from collections import defaultdict

import community as community_louvain
import networkx as nx


def _clusters_from_partition(partition):
    clusters = defaultdict(list)
    for word, cluster_id in partition.items():
        clusters[cluster_id].append(word)
    return clusters


def louvain_method(G):
    if G.number_of_nodes() == 0:
        return defaultdict(list), {}

    partition = community_louvain.best_partition(G, weight="weight", random_state=42)
    clusters = _clusters_from_partition(partition)
    return clusters, partition


def girvan_newman_method(G, level=1):
    if G.number_of_nodes() == 0:
        return defaultdict(list), {}

    if G.number_of_edges() == 0:
        partition = {node: i for i, node in enumerate(G.nodes())}
        return _clusters_from_partition(partition), partition

    comp = nx.community.girvan_newman(G)
    communities = None
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
    if method == "girvan_newman":
        return girvan_newman_method(G)
    raise ValueError("Unknown method")
