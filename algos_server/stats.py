import json
import logging
from typing import Any, Dict

from .algo import FederatedServerAlgo

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

class StatsFederatedServerAlgo(FederatedServerAlgo):

    def __init__(self, params: Dict[str, Any]):
        super().__init__(name="stats", params=params, model_suffix="json")

    def initialize(self):
        # parameters to be passed to the worker
        self.params["cutoff"] = self.params.get("cutoff", 730)
        self.params["delta"] = self.params.get("delta", 30)
        logger.debug("Parameters to be passed to the worker have been initialized: %s", self.params)

        # we definately don't need an initial model for this algorithm but
        # server makes workers start training by sharing an initial model so, we
        # create an empty file here...
        # Hopefully it's clear by now that this whole thing is just a PoC.
        # This is not an actual proper FL implementation with IDS/TSG!
        with open(self.model_aggregated_path, "w+") as f:
            f.write("")

    def aggregate(self, current_round):
        # no real aggregation, just concatenate all partial results
        aggregated_results = [
            json.load(open(file))
            for file in self.round_partial_models[current_round]
        ]
        logger.info("Concatenated %s partial results", len(aggregated_results))
        logger.info("Saving final compiled stats")
        with open(self.model_aggregated_path, "w+") as f:
            json.dump(aggregated_results, f)

        # we only require one round for this algorithm, so aggregation happens
        # only once and we can signal we are done
        return True
