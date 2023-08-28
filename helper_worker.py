import re
import json
import logging
import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype
from scipy.spatial import distance
from typing import Optional, List

logger = logging.getLogger("helper_worker_logger")
data: Optional[np.ndarray] = None
centroids: List = None


def initialize(**kwargs):
    global data
    global centroids

    key = kwargs.get("key")
    centroids = kwargs.get("centroids")

    print(f"Reading data from {key}.csv")
    data = pd.read_csv(f"{key}.csv")
    print(f"Data shape: {data.shape}")

    # Select columns and drop rows with NaNs
    columns = kwargs.get("columns")
    data = data[columns].dropna(how='any')

    # Convert from categorical to numerical TNM, if necessary, values such as
    # Tx, Nx, Mx are converted to -1
    for col in columns:
        if is_string_dtype(data[col]):
            data[col] = data[col].apply(lambda x: re.compile(r'\d').findall(x))
            data[col] = data[col].apply(lambda x: int(x[0]) if len(x) != 0 else -1)

    print("Data has been prepared for clustering")


def train(key: str, starting_centroids: str, k: int, columns: list):
    global data
    global centroids

    print("Start clustering...")
    if starting_centroids is not None:
        print("Loading centroids")
        centroids = json.load(open(starting_centroids))['centroids']

    print("Calculating distance matrix")
    distances = np.zeros([len(data), k])
    for i in range(len(data)):
        for j in range(k):
            xi = list(data.iloc[i].values)
            xj = centroids[j]
            distances[i, j] = distance.euclidean(xi, xj)

    print("Calculating local membership matrix")
    membership = np.zeros([len(data), k])
    for i in range(len(data)):
        j = np.argmin(distances[i])
        membership[i, j] = 1

    print("Generating local cluster centroids")
    centroids = []
    for i in range(k):
        members = membership[:, i]
        dfc = data.iloc[members == 1]
        centroid = []
        for column in columns:
            centroid.append(dfc[column].mean())
        centroids.append(centroid)

    print("Saving local centroids")
    with open(f"centroids.{key}.json", "w+") as f:
        json.dump({"centroids": centroids}, f)
