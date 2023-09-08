import asyncio
import json
import logging
import os
import traceback
from enum import Enum, auto
from threading import Thread

from aiohttp import ClientSession, web
from aiohttp.abc import Request

from algos_server.algo import FederatedServerAlgo
from algos_server.kmeans import KmeansFederatedServerAlgo
from algos_server.stats import StatsFederatedServerAlgo
from algos_server.nn import NNFederatedServerAlgo


debug_sleep_time = int(os.environ.get("DEBUG_SLEEP_TIME", "10"))


def start_background_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()

# will be passed to run_coroutine_threadsafe as a callback, useful for debugging
def done_callback(future: asyncio.Future) -> None:
    """Callback for run_coroutine_threadsafe. Logs exceptions from scheduled coroutines."""
    try:
        result = future.result()
        # TODO: what does this print?
        logger.debug("Scheduled coroutine finished: %s", result)
    except Exception as exception:
        logger.error("Caught exception from scheduled coroutine: %s", exception)
        traceback.print_exc()

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

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self.algo: FederatedServerAlgo = None
        self._state = FederatedLearningState.NONE
        self._workers: list = []
        self._params: dict = {}
        self._rounds: int = 0
        self._current_round: int = 0
        t = Thread(target=start_background_loop, args=(self._loop,), daemon=True)
        t.start()

    # TODO: How do you reach this endpoint, is this a thing in a federated data-app?
    async def status(self, _: Request) -> web.Response:
        logger.info("Status requested")
        return web.json_response(
            data = {
                "rounds": self._rounds,
                "workers": self._workers,
                "current_round": self._current_round,
                "state": self._state.name,
            }
        )

    async def common_model(self, _: Request) -> web.StreamResponse:
        """Returns complete model once the algorithm is finished. And a status json if not finished."""
        logger.info("Model requested")
        response = None
        # FIXME: This is kludgy! But since /status is not apperately reachable (from client?)
        # after hopping through the data-app, we have to resort to this....
        # .. and because from the client we are POSTing (or due to data-app), I'm unable to get query params through.
        # .. and headers neither (not even Content-Type)
        # .. and I can't seem to send headers back either (server -> client)
        # .. 200..299 status codes seem to end up as 200 on the client
        # .. but 400..499 do seem to get forwarded to the client (not always mapped to '400')!
        # .... but nevermind, if it's any 400 it will not let the body through ({'state': self._state.name ...})
        # ..... code is hard to follow, but it might be here?
        #       https://gitlab.com/tno-tsg/data-apps/federated-learning/-/blob/master/src/main/java/nl/tno/ids/fl/trusted/TrustedComputeNodeController.kt#L110
        # OK, I give up, we just say:
        #   425 means model not ready (code not meant for this, but alas..)
        #   200 (OK) means model is ready and you get it in the body
        # [Obviously, this is not how it should be done, but this is a good as we can get without modifying the data-app for now I guess]
        if self.algo and self.algo.get_model_aggregated_data() and self._state == FederatedLearningState.FINISHED:
            response = web.FileResponse(self.algo.get_model_aggregated_path(), status=200)
        else:
            # 'reason' will not make it through data-app...
            response = web.Response(status=400 + 25, reason="Model not ready yet")
        return response

    async def initialize(self, request: Request) -> web.Response:
        self._params = await request.json()
        self._workers = self._params.pop("workers")
        self._rounds = self._params["rounds"]
        self._current_round = 0
        self._state = FederatedLearningState.INITIALIZED

        if self._params.get("algo", None) == "kmeans":
            logger.info("Initializing Kmeans Federated Server Algo")
            self.algo = KmeansFederatedServerAlgo(params=self._params)
        elif self._params.get("algo", None) == "stats":
            logger.info("Initializing Stats collection")
            self.algo = StatsFederatedServerAlgo(params=self._params)
        elif self._params.get("algo", None) == "nn" or self._params.get("algo", None) is None:
            # researcher-gui does not send 'algo', but it's meant for NN
            logger.info("Initializing NN Federated Server Algo")
            self.algo = NNFederatedServerAlgo(params=self._params)
        else:
            logger.warning("Unknown algorithm %s", self._params.get('algo', None))
            return web.Response(status=500)

        logger.info("Initialized Federated Learning for algorithm %s and file %s", self.algo.get_name(), self._params['key'])
        logger.info("Parameters: %s", self.algo.get_params())

        future = asyncio.run_coroutine_threadsafe(self.share_init(), self._loop)
        future.add_done_callback(done_callback)

        return web.Response()

    async def train(self, request: Request) -> web.Response:
        self._state = FederatedLearningState.WAITING

        logger.info("Start training")

        future = asyncio.run_coroutine_threadsafe(self.share_model(), self._loop)
        future.add_done_callback(done_callback)

        return web.Response()
 
    # receive model (raw) from workers
    async def receive_model(self, request: Request) -> web.Response:
        worker = request.headers["Forward-Sender"]

        num_partials = await self.algo.receive_partial_model(request, worker, self._current_round)
        logger.info("Received model from %s", worker)
        logger.info("Partials received so far: %s", num_partials)

        # if we've received all models, aggregate them
        if num_partials == len(self._workers):
            logger.info("Received all models, starting aggregation...")
            await self.aggregate()
        elif num_partials > len(self._workers):
            logger.error("Received more models than workers? %s > %s", num_partials, len(self._workers))

        return web.Response()

    # shares parameters with worksers with workers
    async def share_init(self) -> None:
        logger.info("Sharing init parameters with workers")
        async with ClientSession() as session:
            for worker in self._workers:
                logger.info("Sharing init parameters with %s via %s", worker, f"{data_app_url}/init")
                async with session.post(
                    f"{data_app_url}/init",
                    json=self.algo.get_params(),
                    headers={"Forward-To": worker},
                ) as response:
                    if response.status > 299:
                        logger.error("Error in sharing init parameters with %s: %s", worker, response.status)
                    else:
                        logger.info("Shared init parameters with %s successfully", worker)

    # shares model (raw) with workers, which will in turn receive and start training
    async def share_model(self) -> None:
        self._state = FederatedLearningState.WAITING

        logger.info("Sharing model with workers")
        async with ClientSession() as session:
            for worker in self._workers:
                logger.info("Sharing model with %s via %s", worker, f"{data_app_url}/model")
                async with session.post(
                    f"{data_app_url}/model",
                    data=self.algo.get_model_aggregated_data(),
                    headers={"Forward-To": worker},
                ) as response:
                    logger.info("Received response from %s: %s", worker, response.status)
                    if response.status > 299:
                        logger.error("Error in sharing model %s with %s: %s", self.algo.get_model_aggregated_path(), worker, response.status)
                    else:
                        logger.info("Shared model with %s successfully", worker)

    async def share_finish(self) -> None:
        self._state = FederatedLearningState.FINISHED
        async with ClientSession() as session:
            for worker in self._workers:
                async with session.post(
                    f"{data_app_url}/finish", headers={"Forward-To": worker}
                ) as response:
                    if response.status > 299:
                        logger.error("Error in sharing finish signal with %s: %s", worker, response.status)
                    else:
                        logger.info("Shared finish signal with %s successfully", worker)

    async def aggregate(self) -> None:
        self._state = FederatedLearningState.AVERAGING
        # aggregate(rounds) will return False when the algorithm is finished
        if self._current_round < self._rounds:
            finished = self.algo.aggregate(self._current_round)
        else:
            logger.info("Reached maximum number of rounds")
            finished = True

        if finished:
            logger.info("Federated Learning finished")
            await self.share_finish()
        else:
            logger.info("Federated Learning not finished yet, sharing model..")
            await self.share_model()

        self._current_round += 1


if __name__ == "__main__":
    # Logging setup
    FORMAT = "%(asctime)s - %(name)s | %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)

    if os.environ.get("DATA_APP_URL") is None:
        logger.warning("DATA_APP_URL not set, will use default!")
    data_app_url = os.environ.get("DATA_APP_URL", "http://localhost:8085")

    federated_learning_server_handler = FederatedLearningServerHandler()

    app = web.Application(client_max_size=1024**3)
    app.router.add_post("/initialize", federated_learning_server_handler.initialize)
    app.router.add_post("/train", federated_learning_server_handler.train)
    # 'post /model' to the data-app of the worker gets results in a POST here
    app.router.add_post("/model", federated_learning_server_handler.receive_model)
    # 'post /model' to the data-app of the server results in a GET here.
    app.router.add_get("/model", federated_learning_server_handler.common_model)
    # Unsure how to actually reach this as a client..
    app.router.add_get("/status", federated_learning_server_handler.status)
    web.run_app(app, host="0.0.0.0", port=8080)
