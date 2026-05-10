import networkx as nx


def get_top_words(G, clusters, top_n=5):
    result = {}

    for cid, words in clusters.items():
        scored = sorted(
            words,
            key=lambda w: G.degree(w, weight="weight"),
            reverse=True,
        )
        result[cid] = scored[:top_n]

    return result


def summarize_clusters(G, clusters):
    cluster_sizes = sorted((len(words) for words in clusters.values()), reverse=True)
    summary = {
        "cluster_count": len(clusters),
        "largest_cluster": cluster_sizes[0] if cluster_sizes else 0,
        "smallest_cluster": cluster_sizes[-1] if cluster_sizes else 0,
        "avg_cluster_size": round(sum(cluster_sizes) / len(cluster_sizes), 2)
        if cluster_sizes
        else 0,
    }

    if G.number_of_edges() > 0 and clusters:
        communities = [set(words) for words in clusters.values() if words]
        summary["modularity"] = round(
            nx.community.modularity(G, communities, weight="weight"),
            4,
        )
    else:
        summary["modularity"] = 0

    return summary
