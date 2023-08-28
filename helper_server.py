import json
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, Any, List


def initialize(model_centroids: str, **kwargs) -> Dict[str, Any]:
    # Centroids
    centroids_input = kwargs.get("centroids", None)
    if centroids_input is None:
        # These are the initial centroids if none are given, it only makes
        # sense for the TNM data of the PoC
        kwargs["centroids"] = [[1, 0, 0], [2, 1, 0], [3, 2, 0], [5, 3, 1]]

    # Number of clusters k
    k_input = kwargs.get("k", None)
    if k_input is None:
        kwargs["k"] = 4

    # Convergence criterion epsilon
    epsilon_input = kwargs.get("epsilon", None)
    if epsilon_input is None:
        kwargs["epsilon"] = 0.01

    # Maximum number of iterations
    max_iter_input = kwargs.get("max_iter", None)
    if max_iter_input is None:
        kwargs["max_iter"] = 50

    # Columns to use for clustering
    columns_input = kwargs.get("columns", None)
    if columns_input is None:
        # TODO: use all columns from input file when none are given
        kwargs["columns"] = ['t', 'n', 'm']

    # Change in centroids, initialize as something bigger than epsilon
    change = 2*kwargs["epsilon"]
    kwargs["change"] = change

    # Save initial centroids
    print("Saving initial centroids")
    with open(model_centroids, "w+") as f:
        json.dump(kwargs["centroids"], f)

    return kwargs


def average_weights(model_centroids: str, full_model: str, files: List[str], k: int) -> None:
    print(f"Averaging {files} into {model_centroids}")
    results = [json.load(open(file))['centroids'] for file in files]
    print("Loaded centroids")

    # Organise local centroids into a matrix
    local_centroids = []
    for result in results:
        for local_centroid in result:
            local_centroids.append(local_centroid)
    X = np.array(local_centroids)

    # Average centroids by running kmeans on local results
    print('Run global averaging for centroids')
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    new_centroids = kmeans.cluster_centers_

    # Read previous centroids
    centroids = json.load(open(model_centroids))['centroids']

    # Compute the sum of the magnitudes of the centroids differences
    # between steps. This change in centroids between steps will be used
    # to evaluate convergence.
    print('Compute change in cluster centroids')
    change = 0
    for i in range(k):
        diff = new_centroids[i] - np.array(centroids[i])
        change += np.linalg.norm(diff)

    # Re-define the centroids
    centroids = {'centroids': list(list(centre) for centre in new_centroids)}

    # Save results
    print("Saving average centroids")
    with open(model_centroids, "w+") as f:
        json.dump(centroids, f)
    with open(full_model, "w+") as f:
        json.dump(centroids, f)

    return change
