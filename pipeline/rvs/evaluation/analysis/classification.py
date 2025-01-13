from typing import Dict

import numpy as np
from numpy.typing import NDArray

from rvs.evaluation.analysis.utils import Method
from rvs.evaluation.lvis import Category, Uid


def classify(embedding: NDArray, class_embeddings: Dict[Category, NDArray]) -> Category:
    assert len(class_embeddings) > 0
    best_similarity = -1.0
    best_class: Category = None
    for query, query_embedding in class_embeddings:
        similarity = np.dot(embedding, query_embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_class = query
    assert best_class is not None
    return best_class


def classify_all(
    embeddings: Dict[Method, Dict[Uid, NDArray]],
    class_embeddings: Dict[Category, NDArray],
) -> Dict[Method, Dict[Uid, Category]]:
    predictions: Dict[Method, Dict[Uid, Category]] = dict()

    for method in embeddings.keys():
        method_embeddings = embeddings[method]

        method_predictions: Dict[Uid, Category] = dict()

        for uid in method_embeddings.keys():
            method_predictions[uid] = classify(method_embeddings[uid], class_embeddings)

        predictions[method] = method_predictions

    return predictions
