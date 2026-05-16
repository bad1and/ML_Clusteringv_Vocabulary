from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from config import OUTPUT_DIR, SAVE_PLOTS, SHOW_PLOTS


def draw_graph(G, partition, title="Graph", output_name="graph.png"):
    if G.number_of_nodes() == 0:
        print("Graph is empty, nothing to draw.")
        return

    pos = nx.spring_layout(G, seed=42, weight="weight", k=0.8, iterations=80)

    colors = [partition.get(node, 0) for node in G.nodes()]
    raw_weights = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
    widths = [0.5 + 4 * w for w in raw_weights]

    plt.figure(figsize=(13, 9))
    plt.title(title)

    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.35)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=550, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.axis("off")
    plt.tight_layout()

    if SAVE_PLOTS:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = Path(OUTPUT_DIR) / output_name
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved graph plot to: {output_path}")

    if SHOW_PLOTS:
        plt.show(block=False)

    plt.close()
