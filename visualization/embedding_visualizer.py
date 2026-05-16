from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from config import OUTPUT_DIR, SAVE_PLOTS, SHOW_PLOTS


def visualize_embeddings(words, embeddings, title="Sentence-BERT embeddings", output_name="embeddings.png"):
    if len(words) < 2:
        print("Not enough words to visualize embeddings.")
        return

    pca = PCA(n_components=2, random_state=42)
    points = pca.fit_transform(embeddings)

    plt.figure(figsize=(14, 9))

    for i, word in enumerate(words):
        x, y = points[i]
        plt.scatter(x, y)
        plt.text(x + 0.01, y + 0.01, word, fontsize=8)

    plt.title(title)
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.grid(True)

    if SAVE_PLOTS:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = Path(OUTPUT_DIR) / output_name
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved embedding plot to: {output_path}")

    if SHOW_PLOTS:
        plt.show(block=False)
        plt.pause(0.1)

    plt.close()
