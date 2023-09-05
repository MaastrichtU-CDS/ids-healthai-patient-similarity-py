import re
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import List

import aiofiles
from aiohttp.abc import Request


class FederatedServerAlgo(object):
    _file_pattern = re.compile("[\\W_]+")

    def __init__(self, tmp_dir: Path = Path("./tmp"), name: str = "general", params: dict = {}, model_suffix: str = "data"):
        # name of algorithm
        self.name = name
        # path under which to store models
        self.models_dir = tmp_dir / self.name
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_suffix = model_suffix
        # path to aggregated model
        self.model_aggregated_path = self.models_dir / f"model-aggregated.{self.model_suffix}"
        # keep track of how many partials per round
        self.round_partial_models = defaultdict(list)
        # validate key is sane filename
        if not re.match("^[a-zA-Z0-9_.]{1,32}$", params["key"]):
            raise ValueError("Invalid name for key")
        # params to be shared with workers
        self.params = params
        self.initialize()

    def get_name(self):
        return self.name

    def get_params(self):
        return self.params

    def get_model_aggregated_path(self):
        return self.model_aggregated_path

    def get_model_aggregated_data(self):
        with open(self.model_aggregated_path, "rb") as f:
            return f.read()

    def get_partial_files(self, round) -> List[Path]:
        return self.round_partial_models[round]

    async def receive_partial_model(self, request: Request, worker, current_round):
        # create file for partial model from a worker
        file_path = self.models_dir / f"model-partial-{self._file_pattern.sub('', worker)}-{current_round}.{self.model_suffix}"
        with open(file_path, "w+") as f:
            f.write("")

        # write partial model to file
        async with aiofiles.open(file_path, "ba+") as f:
            async for data in request.content.iter_chunked(10240):
                await f.write(data)

        # add file to list of partial models for this round
        self.round_partial_models[current_round].append(file_path)

        # return how many partial models we have so far for this round
        return len(self.round_partial_models[current_round])

    @abstractmethod
    def initialize(self):
        """Initializes the algorithm and sets parameters to be shared with workers"""
    
    @abstractmethod
    def aggregate(self, current_round):
        """Aggregates received partial models from workers for the given round"""
    