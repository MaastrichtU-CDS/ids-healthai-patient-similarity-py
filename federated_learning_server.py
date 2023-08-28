import asyncio
import json
import logging
import os
from enum import Enum, auto
from pathlib import Path
from threading import Thread

import aiofiles
import re
from aiohttp import web, ClientSession
from aiohttp.abc import Request
from random import randrange

import helper_server

data_app_url = os.environ.get("DATA_APP_URL", "https://httpbin.org/anything")
debug_sleep_time = int(os.environ.get("DEBUG_SLEEP_TIME", "10"))


def start_background_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


class FederatedLearningState(Enum):
    """
    State enum for indicating the state of the Federated Learning server
    """

    NONE = auto()
    INITIALIZED = auto()
    WAITING = auto()
    AVERAGING = auto()
    FINISHED = auto()


class FederatedLearningServerHandler:
    """
    Federated Learning Server Handler
    """

    _state = FederatedLearningState.NONE
    _tmp_models = "./tmp"
    _params: dict = {}
    _workers: list = []
    _rounds: int = 0
    _current_round: int = 0
    _received_models: int = 0
    _file_pattern = re.compile("[\\W_]+")

    def __init__(self):
        Path(self._tmp_models).mkdir(parents=True, exist_ok=True)
        self._loop = asyncio.new_event_loop()
        t = Thread(target=start_background_loop, args=(self._loop,), daemon=True)
        t.start()

    async def status(self, _: Request) -> web.Response:
        return web.json_response(
            data={
                "rounds": self._rounds,
                "workers": self._workers,
                "current_round": self._current_round,
                "state": self._state.name,
            }
        )

    async def common_model(self, _: Request) -> web.StreamResponse:
        return web.FileResponse(f"{self._tmp_models}/full_model.json")

    async def initialize(self, request: Request) -> web.Response:
        self._params = await request.json()
        self._workers = self._params.pop("workers")
        self._rounds = self._params["rounds"]
        self._state = FederatedLearningState.INITIALIZED
        self._params = helper_server.initialize(
            f"{self._tmp_models}/common_model.json", **self._params
        )
        self._params["parties"] = len(self._workers)
        self._params["name"] = f'{self._params["key"]}-cnn-{randrange(100000,999999)}'
        logger.info("Initialized Federated Learning")
        logger.info(self._params)
        asyncio.run_coroutine_threadsafe(self.share_init(), self._loop)
        return web.Response()

    async def train(self, request: Request) -> web.Response:
        self._state = FederatedLearningState.WAITING

        logger.info("Start training")

        asyncio.run_coroutine_threadsafe(self.share_model(), self._loop)
        return web.Response()

    async def receive_model(self, request: Request) -> web.Response:
        worker = request.headers["Forward-Sender"]
        file_path = f'{self._tmp_models}/{self._file_pattern.sub("", worker)}_{self._current_round}.json'
        with open(file_path, "w+") as f:
            json.dump("", f)

        async with aiofiles.open(file_path, "a+") as f:
            # TODO: might need to change something here, as we are writing a simple json
            async for data in request.content.iter_chunked(10240):
                await json.dump(data, f)

        logger.info(f"Received model from {worker}")
        self._received_models += 1
        if self._received_models >= len(self._workers):
            await self.calculate_average()

        return web.Response()

    async def share_init(self) -> None:
        async with ClientSession() as session:
            for worker in self._workers:
                async with session.post(
                    f"{data_app_url}/init",
                    json=self._params,
                    headers={"Forward-To": worker},
                ) as response:
                    if response.status > 299:
                        logger.error(
                            f"Error in sharing init parameters with {worker}: {response.status}"
                        )
                    else:
                        logger.info(
                            f"Shared init parameters with {worker} successfully"
                        )

    async def share_model(self) -> None:
        self._state = FederatedLearningState.WAITING
        async with ClientSession() as session:
            for worker in self._workers:
                async with session.post(
                    f"{data_app_url}/model",
                    data=json.load(open(f"{self._tmp_models}/common_model.json")),
                    headers={"Forward-To": worker},
                ) as response:
                    if response.status > 299:
                        logger.error(
                            f"Error in sharing model with {worker}: {response.status}"
                        )
                    else:
                        logger.info(f"Shared model with {worker} successfully")

    async def share_finish(self) -> None:
        self._state = FederatedLearningState.FINISHED
        async with ClientSession() as session:
            for worker in self._workers:
                async with session.post(
                    f"{data_app_url}/finish", headers={"Forward-To": worker}
                ) as response:
                    if response.status > 299:
                        logger.error(
                            f"Error in sharing finish signal with {worker}: {response.status}"
                        )
                    else:
                        logger.info(f"Shared finish signal with {worker} successfully")

    async def calculate_average(self) -> None:
        self._state = FederatedLearningState.AVERAGING
        files = [
            (f'{self._tmp_models}/{self._file_pattern.sub("", worker)}_{self._current_round}.json')
            for worker in self._workers
        ]
        output = f"{self._tmp_models}/common_model.json"
        full_output = f"{self._tmp_models}/full_model.json"

        self._current_round += 1
        self._received_models = 0
        logger.info(f"Calculating average of {files} into {output}")

        self._params["change"] = helper_server.average_weights(output, full_output, files, self._params["k"])

        # Stop when maximum number of iterations or convergence criterion is achieved
        if (self._current_round > self._params["max_iter"]) or (self._params["change"] <= self._params["epsilon"]):
            await self.share_finish()
        else:
            await self.share_model()


if __name__ == "__main__":
    # Logging setup
    FORMAT = "%(asctime)s | %(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger("worker")
    logger.setLevel(level=logging.DEBUG)

    federated_learning_server_handler = FederatedLearningServerHandler()

    app = web.Application(client_max_size=1024**3)
    app.router.add_post("/initialize", federated_learning_server_handler.initialize)
    app.router.add_post("/train", federated_learning_server_handler.train)
    app.router.add_post("/model", federated_learning_server_handler.receive_model)
    app.router.add_get("/status", federated_learning_server_handler.status)
    app.router.add_get("/model", federated_learning_server_handler.common_model)
    web.run_app(app, host="0.0.0.0", port=8080)
