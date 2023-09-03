import json
import logging
import re
from typing import List, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from scipy.spatial import distance

from .algo import FederatedWorkerAlgo

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

class KmeansFederatedWorkerAlgo(FederatedWorkerAlgo):
    def __init__(self, params: dict = {}):
        self.data: Optional[np.ndarray] = None
        self.centroids: List = None
        super().__init__(name="kmeans", params=params, model_suffix="json")

    def initialize(self):
        self.k = self.params.get("k")
        self.centroids = self.params.get("centroids")
        self.columns = self.params.get("columns")

        logger.info("Reading data from %s.csv", self.key)
        self.data = pd.read_csv(f"{self.key}.csv")
        logger.info("Data shape: %s", self.data.shape)

        # Select columns and drop rows with NaNs
        self.data = self.data[self.columns].dropna(how='any')

        # Convert from categorical to numerical TNM, if necessary, values such as
        # Tx, Nx, Mx are converted to -1
        for col in self.columns:
            if is_string_dtype(self.data[col]):
                self.data[col] = self.data[col].apply(lambda x: re.compile(r'\d').findall(x))
                self.data[col] = self.data[col].apply(lambda x: int(x[0]) if len(x) != 0 else -1)

        logger.info("Data has been prepared for clustering")

    def train(self, callback=None):
        logger.info("Start clustering...")
        if self.pre_model_path is not None:
            logger.info("Loading centroids")
            self.centroids = json.load(open(self.pre_model_path))['centroids']

        logger.info("Calculating distance matrix")
        distances = np.zeros([len(self.data), self.k])
        for i in range(len(self.data)):
            for j in range(self.k):
                xi = list(self.data.iloc[i].values)
                xj = self.centroids[j]
                distances[i, j] = distance.euclidean(xi, xj)

        logger.info("Calculating local membership matrix")
        membership = np.zeros([len(self.data), self.k])
        for i in range(len(self.data)):
            j = np.argmin(distances[i])
            membership[i, j] = 1

        logger.info("Generating local cluster centroids")
        centroids = []
        for i in range(self.k):
            members = membership[:, i]
            dfc = self.data.iloc[members == 1]
            centroid = []
            for column in self.columns:
                centroid.append(dfc[column].mean())
            centroids.append(centroid)

        logger.info("Saving local centroids")
        with open(self.model_path, "w+") as f:
            json.dump({"centroids": centroids}, f)

        # TODO: return something meaningful. Althought we are mostly interested
        # in the global (server side) status, which at the moment we seem to be
        # unable to reach.
        callback({"staus": "in-progress"})

