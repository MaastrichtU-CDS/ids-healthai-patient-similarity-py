import logging
from typing import List, Optional, Any, Dict, Callable, Tuple

import json
import keras.models
import keras.callbacks
import pandas as pd

from numpy import ndarray
import tensorflow as tf
import numpy as np

logger = logging.getLogger("helper_worker_logger")

data: Optional[ndarray] = None
labels: Optional[ndarray] = None
cluster: Optional[int] = None
columns: Optional[List[str]] = None
# unique_labels: Optional[int] = None
# array_features_dict: dict = {}


def initialize(**kwargs):
    global data
    # global labels
    global clusters
    global columns

    key = kwargs.get("key")

    clusters = kwargs.get("clusters", 4)
    columns = kwargs.get("columns", [])

    print(f"Reading data from {key}.csv")
    data = pd.read_csv(f"{key}.csv")
    print(f"Data shape: {data.shape}")

    # features_dict = data.copy()
    # for name, column in features_dict.items():
    #     dtype = column.dtype
    #     if dtype == object:
    #         features_dict[name] = features_dict[name].fillna("")

    # labels = data[label_column]
    # labels = pd.get_dummies(labels)

    # array_features_dict = {
    #     name: np.array(value) for name, value in features_dict.items()
    # }


def train(
    key: str,
    starting_weights: str,
    epochs: int,
    callback: Callable[[Dict[str, Any]], None],
    batch_size=8,
    validation_split=0.1,
) -> Tuple[str, str, str, str]:
    global data
    global labels
    global columns
    global clusters
    # global features_dict
    print(
        f"Start training ({key}, {starting_weights}, {epochs}, {batch_size}, {validation_split})"
    )


    with open(starting_weights, 'r') as openfile:
        centroids = json.load(openfile)

    # KMeans on centroids and data

    json_object = json.dumps(centroids)
    
    with open(f"weights.{key}.h5", "w") as outfile:
        outfile.write(json_object)

    return True

