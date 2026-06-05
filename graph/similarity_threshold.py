import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple


def compute_all_similarities(embeddings: np.ndarray) -> np.ndarray:

    sim_matrix = cosine_similarity(embeddings)

    # берём верхний треугольник без диагонали
    triu_indices = np.triu_indices_from(sim_matrix, k=1)
    all_sims = sim_matrix[triu_indices]
    return all_sims


def get_similarity_threshold(
        embeddings: np.ndarray,
        percentile: float = 85.0,
        min_threshold: float = 0.15,
        max_threshold: float = 0.9
) -> float:

    if len(embeddings) < 2:
        return 0.3  # слишком мало слов для анализа

    all_sims = compute_all_similarities(embeddings)

    if len(all_sims) == 0:
        return 0.3

    threshold = np.percentile(all_sims, percentile)

    # Ограничиваем порог разумными пределами
    threshold = np.clip(threshold, min_threshold, max_threshold)

    return float(threshold)


def get_similarity_stats(embeddings: np.ndarray) -> dict:

    if len(embeddings) < 2:
        return {
            "min": 0.0, "max": 0.0, "mean": 0.0,
            "median": 0.0, "q25": 0.0, "q75": 0.0,
            "num_pairs": 0
        }

    all_sims = compute_all_similarities(embeddings)

    return {
        "min": float(np.min(all_sims)),
        "max": float(np.max(all_sims)),
        "mean": float(np.mean(all_sims)),
        "median": float(np.median(all_sims)),
        "q25": float(np.percentile(all_sims, 25)),
        "q75": float(np.percentile(all_sims, 75)),
        "num_pairs": len(all_sims),
    }