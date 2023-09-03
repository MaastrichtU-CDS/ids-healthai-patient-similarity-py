import asyncio
import logging
import os
import traceback
from enum import Enum
from threading import Thread
from typing import Any, Dict

import requests
from aiohttp import ClientSession, web
from aiohttp.abc import Request

from algos_worker.algo import FederatedWorkerAlgo
from algos_worker.kmeans import KmeansFederatedWorkerAlgo
from algos_worker.nn import NNFederatedWorkerAlgo
from dataset_handler import DataSetHandler

debug_sleep_time = int(os.environ.get("DEBUG_SLEEP_TIME", "10"))


def start_background_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()

# will be passed to run_coroutine_threadsafe as a callback, useful for debugging
def done_callback(future: asyncio.Future) -> None:
    """Callback for run_coroutine_threadsafe. Logs exceptions from scheduled coroutines."""
    try:
        result = future.result()
    except Exception as exception:
        logger.error("Caught exception from scheduled coroutine: %s", exception)
        traceback.print_exc()

class FederatedLearningState(str, Enum):
    NONE = "NONE"
    INITIALIZED = "INITIALIZED"
    WAITING = "WAITING"
    TRAINING = "TRAINING"
    FINISHED = "FINISHED"


class FederatedLearningHandler:
    def __init__(self):
        self.algo: FederatedWorkerAlgo = None
        self._state = FederatedLearningState.NONE
        self._params: Dict[str, Any] = {}
        self._status = []
        self._round = -1
        self._loop = asyncio.new_event_loop()
        t = Thread(target=start_background_loop, args=(self._loop,), daemon=True)
        t.start()

    async def status(self, request: Request) -> web.Response:
        return web.json_response({"state": self._state, "status": self._status})

    async def initialize(self, request: Request) -> web.Response:
        self._params = await request.json()
        self._state = FederatedLearningState.INITIALIZED
        self._round = 0

        if self._params.get("algo") == "kmeans":
            logger.info("Initializing Kmeans Federated Learning")
            self.algo = KmeansFederatedWorkerAlgo(params=self._params)
        else:
            logger.info("Initializing NN Federated Learning")
            self.algo = NNFederatedWorkerAlgo(params=self._params)

        logger.info("Initialized Federated Learning")
        logger.info(self._params)
        return web.Response()

    # TODO: naming of these could be improved
    # train() @ server
    #  share_model() @ server
    #   data-app/model ->
    #    train() @ worker
    async def train(self, request: Request) -> web.Response:
        self._state = FederatedLearningState.TRAINING

        # extract model from request and store it
        await self.algo.receive_model(request)

        logger.info("Received input model, start training")
        future = asyncio.run_coroutine_threadsafe(self.train_model(), self._loop)
        future.add_done_callback(done_callback)

        return web.Response()

    async def finish(self, _: Request) -> web.Response:
        logger.info("Federated Learning finished")
        self._state = FederatedLearningState.FINISHED
        return web.Response()

    async def share_model(self) -> None:
        self._state = FederatedLearningState.WAITING
        async with ClientSession() as session:
            async with session.post(
                f"{data_app_url}/model", data=self.algo.get_model_data()
            ) as response:
                if response.status > 299:
                    logger.error("Error in sharing model: %s", response.status)
                else:
                    logger.info("Shared model successfully")

    def share_status(self, status):
        logger.info("Sharing status: %s", status)
        status_round = {**status, "round": self._round}
        self._status.append(status_round)
        response = requests.post(f"{data_app_url}/status", json=status_round)
        if response.status_code > 299:
            logger.error("Error in sharing status: %s", response.status_code)
        else:
            logger.info("Shared status successfully")

    async def train_model(self) -> None:
        logger.info("Training..")
        logger.info(self.algo.params)
        self.algo.train(callback=self.share_status)
        logger.info("Training finished")
        self._round += 1
        await self.share_model()


if __name__ == "__main__":
    # Logging setup
    FORMAT = "%(asctime)s - %(name)s | %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    if os.environ.get("DATA_APP_URL") is None:
        logger.warning("DATA_APP_URL not set, will use default!")
    data_app_url = os.environ.get("DATA_APP_URL", "http://localhost:8085")

    federated_learning_handler = FederatedLearningHandler()
    dataset_handler = DataSetHandler()
    # asyncio.get_event_loop().create_task()
    app = web.Application(client_max_size=1024**3)
    app.router.add_post("/initialize", federated_learning_handler.initialize)
    app.router.add_post("/model", federated_learning_handler.train)
    app.router.add_post("/finish", federated_learning_handler.finish)
    app.router.add_get("/status", federated_learning_handler.status)

    app.router.add_post("/datasets/{key}", dataset_handler.add_dataset)
    app.router.add_delete("/datasets/{key}", dataset_handler.delete_dataset)
    app.router.add_get("/datasets", dataset_handler.list_datasets)

    web.run_app(app, host="0.0.0.0", port=8080)
