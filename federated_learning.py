import asyncio
import logging
import os
from enum import Enum
from pathlib import Path
from threading import Thread
from typing import Any, Dict

import aiofiles
import requests
from aiohttp import web, ClientSession
from aiohttp.abc import Request

import helper_worker
from dataset_handler import DataSetHandler

data_app_url = os.environ.get("DATA_APP_URL", "https://httpbin.org/anything")
debug_sleep_time = int(os.environ.get("DEBUG_SLEEP_TIME", "10"))


def start_background_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


class FederatedLearningState(str, Enum):
    NONE = "NONE"
    INITIALIZED = "INITIALIZED"
    WAITING = "WAITING"
    TRAINING = "TRAINING"
    FINISHED = "FINISHED"


class FederatedLearningHandler:
    _state = FederatedLearningState.NONE
    _tmp_model: str = "./tmp/intermediate_model.h5"
    _params: Dict[str, Any] = {}
    _status = []
    _round = -1

    def __init__(self):
        Path(self._tmp_model).parent.mkdir(parents=True, exist_ok=True)
        self._loop = asyncio.new_event_loop()
        t = Thread(target=start_background_loop, args=(self._loop,), daemon=True)
        t.start()

    async def status(self, request: Request) -> web.Response:
        return web.json_response({"state": self._state, "status": self._status})

    async def initialize(self, request: Request) -> web.Response:
        self._params = await request.json()
        self._state = FederatedLearningState.INITIALIZED
        self._round = 0
        logger.info("Initialized Federated Learning")
        logger.info(self._params)
        helper_worker.initialize(**self._params)
        return web.Response()

    async def train(self, request: Request) -> web.Response:
        self._state = FederatedLearningState.TRAINING
        with open(self._tmp_model, "w+") as f:
            f.write("")

        async with aiofiles.open(self._tmp_model, "ba+") as f:
            async for data in request.content.iter_chunked(10240):
                await f.write(data)

        logger.info("Received input model, start training")
        asyncio.run_coroutine_threadsafe(self.train_model(), self._loop)
        return web.Response()

    async def finish(self, _: Request) -> web.Response:
        logger.info("Federated Learning finished")
        self._state = FederatedLearningState.FINISHED
        return web.Response()

    async def share_model(self, file: str) -> None:
        self._state = FederatedLearningState.WAITING
        async with ClientSession() as session:
            async with session.post(
                f"{data_app_url}/model", data=open(file, "rb")
            ) as response:
                if response.status > 299:
                    logger.error(f"Error in sharing model: {response.status}")
                else:
                    logger.info("Shared model successfully")

    def share_status(self, status):
        logger.info(f"Sharing status: {status}")
        status_round = {**status, "round": self._round}
        self._status.append(status_round)
        response = requests.post(f"{data_app_url}/status", json=status_round)
        if response.status_code > 299:
            logger.error(f"Error in sharing status: {response.status_code}")
        else:
            logger.info("Shared status successfully")

    async def train_model(self) -> None:
        logger.info("Training..")
        logger.info(self._params)
        logger.info(self._tmp_model)

        train_result = helper_worker.train(
            self._params["key"],
            starting_weights=self._tmp_model,
            epochs=self._params["epochs"],
            callback=self.share_status,
            batch_size=self._params.get("batch_size", 8),
            validation_split=self._params.get("validation_split", 0.1),
        )
        self._round += 1
        await self.share_model(f"weights.{self._params['key']}.h5")


if __name__ == "__main__":
    # Logging setup
    FORMAT = "%(asctime)s | %(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger("worker")
    logger.setLevel(level=logging.DEBUG)

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
