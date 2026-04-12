import community as community_louvain
from collections import defaultdict

def detect_communities(G):
    partition = community_louvain.best_partition(G)

    clusters = defaultdict(list)
    for word, cluster_id in partition.items():
        clusters[cluster_id].append(word)

    return clusters, partition