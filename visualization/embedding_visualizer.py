import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def visualize_embeddings(words, embeddings, title="Sentence-BERT embeddings"):
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
    plt.show(block=False)
    plt.pause(0.1)