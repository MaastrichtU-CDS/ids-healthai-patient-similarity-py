import json
import logging
from typing import Any, Dict

import numpy as np
from sklearn.cluster import KMeans

from .algo import FederatedServerAlgo

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

class KmeansFederatedServerAlgo(FederatedServerAlgo):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(name="kmeans", params=params, model_suffix="json")

    def initialize(self):
        # self.params will be passed to the worker
        # Centroids
        # Note: these are the initial centroids if none are given, it only makes
        #       sense for the TNM data of the PoC
        self.params["centroids"] = self.params.get("centroids", [[1, 0, 0], [2, 1, 0], [3, 2, 0], [5, 3, 1]])
        # Number of clusters k
        self.params["k"] = self.params.get("k", len(self.params["centroids"]))
        # Columns to use for clustering
        # TODO: use all columns from input file when none are given
        self.params["columns"] = self.params.get("columns", ['t', 'n', 'm'])

        # Server will use these, pop'ed as worker doesn't need them
        self.k = self.params["k"]
        self.centroids = self.params["centroids"]
        # Convergence criterion epsilon
        self.epsilon = self.params.pop("epsilon", 0.01)
        # Maximum number of iterations
        self.max_iter = self.params.pop("max_iter", 50)

        # Change in centroids, initialize as something bigger than epsilon
        # TODO: don't think we need this here..
        #self.change = 2*self.epsilon

        # validation
        if self.k != len(self.centroids):
            raise ValueError("Number of intial centroids must be equal to k!")
        # Save initial centroids
        logger.info("Saving initial centroids")
        with open(self.model_aggregated_path, "w+") as f:
            json.dump({"centroids": self.centroids}, f)


    def aggregate(self, current_round):
        logger.info("Averaging %s into %s", self.round_partial_models[current_round], self.model_aggregated_path)
        workers_centroids = [json.load(open(file))['centroids'] for file in self.round_partial_models[current_round]]
        logger.info("Loaded centroids")

        # Organise local centroids into a matrix
        X = np.vstack(workers_centroids)
        # Note: it can happen that a center had no rows that belonged to a
        #       particular cluster, we will get a row of NaNs for the newly
        #       "computed" value of that cluster centroid. So, we:
        # filter out any centroid from any center that has NaNs
        X = X[~np.any(np.isnan(X), axis=1)]

        # Read previous centroids
        centroids = json.load(open(self.model_aggregated_path))['centroids']

        # if there are not enough centroids for the number of clusters, we also use initial centroids
        # TODO: how incorrect is this?
        if len(X) < self.k:
            X = np.vstack([X, centroids])
            logger.info("Not enough centroids from workers (%s), also using initial centroids", len(X))

        # Average centroids by running kmeans on local results
        logger.info('Run global averaging for centroids')
        kmeans = KMeans(n_clusters=self.k, random_state=0).fit(X)
        new_centroids = kmeans.cluster_centers_


        # Compute the sum of the magnitudes of the centroids differences
        # between steps. This change in centroids between steps will be used
        # to evaluate convergence.
        logger.info("Compute change in cluster centroids")
        change = 0
        for i in range(self.k):
            diff = new_centroids[i] - np.array(centroids[i])
            change += np.linalg.norm(diff)
        logger.info("Change in cluster centroids: %s", change)

        # Re-define the centroids
        centroids = {'centroids': list(list(centre) for centre in new_centroids)}

        # Save results
        logger.info("Saving average centroids")
        with open(self.model_aggregated_path, "w+") as f:
            json.dump(centroids, f)

        # return True when a "finished" criterion is met
        # TODO: for nn we're controlling rounds from server handler.. so this is
        #       just another flaw of this refactor....
        return change < self.epsilon or current_round >= self.max_iter
