from typing import Optional, Dict, Any, List

import json

import numpy as np
import pandas as pd

from scipy.spatial import distance
from sklearn.cluster import KMeans
from pandas.api.types import is_string_dtype


def initialize(model_centroids: str, **kwargs) -> Dict[str, Any]:
    global k
    global centroids
    global epsilon
    global max_iter
    global columns
    global change

    # Centroids
    centroids_input = kwargs.get("centroids", None)
    if centroids_input is None:
        # These are the initial centroids if none are given, it only makes
        # sense for the TNM data of the PoC
        centroids = [[1, 0, 0], [2, 1, 0], [3, 2, 0], [5, 3, 1]]
        kwargs["centroids"] = centroids
    else:
        centroids = centroids_input

    # Number of clusters k
    k_input = kwargs.get("k", None)
    if k_input is None:
        k = 4
        kwargs["k"] = k
    else:
        k = k_input

    # Convergence criterion epsilon
    epsilon_input = kwargs.get("epsilon", None)
    if epsilon_input is None:
        epsilon = 0.01
        kwargs["epsilon"] = epsilon
    else:
        epsilon = epsilon_input

    # Maximum number of iterations
    max_iter_input = kwargs.get("max_iter", None)
    if max_iter_input is None:
        max_iter = 50
        kwargs["max_iter"] = max_iter
    else:
        max_iter = max_iter_input

    # Columns to use for clustering
    columns_input = kwargs.get("columns", None)
    if columns_input is None:
        # TODO: use all columns from input file when none are given
        columns = ['t', 'n', 'm']
        kwargs["columns"] = columns
    else:
        columns = columns_input

    # Change in centroids, initialize as something bigger than epsilon
    change = 2*epsilon
    kwargs["change"] = change

    # TODO: in the original code model_weigths seem to be saved, do we need
    #  to save the centroids, a common_model file is given as input here,
    #  but I'm unsure how it is used
    return kwargs


def get_weights(file: str):
    model = json.load(open(file))['centroids']
    return model


def average_weights(model_weights: str, full_model: str, files: List[str]) -> None:
    # TODO: make sure things are properly saved here
    print(f"Averaging {files} into {model_weights}")
    results = [get_weights(file) for file in files]
    print("Loaded weights")

    # Organise local centroids into a matrix
    local_centroids = []
    for result in results:
        for local_centroid in result:
            local_centroids.append(local_centroid)
    X = np.array(local_centroids)

    # Average centroids by running kmeans on local results
    print('Run global averaging for centroids')
    # TODO: get k from input params
    k = 4
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    new_centroids = kmeans.cluster_centers_

    # Compute the sum of the magnitudes of the centroids differences
    # between steps. This change in centroids between steps will be used
    # to evaluate convergence.
    print('Compute change in cluster centroids')
    change = 0
    for i in range(k):
        diff = new_centroids[i] - np.array(model_weights[i])
        change += np.linalg.norm(diff)

    # Re-define the centroids
    model_weights = list(list(centre) for centre in new_centroids)

    # TODO: somehow save the result
    # print("Saving average weights")
    # model.save_weights(model_weights)
    # model.save(full_model)
