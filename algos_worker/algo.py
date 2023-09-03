import logging
from abc import ABC, abstractmethod
from pathlib import Path

import aiofiles
from aiohttp.abc import Request

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

class FederatedWorkerAlgo(object):
    def __init__(self, tmp_dir: Path = Path("./tmp"), name: str = "general", params: dict = {}, model_suffix: str = "data"):
        self.key: str = params.get("key")
        if self.key is None:
            raise ValueError("\'key\' not provided!")
        self.name: str = name
        self.pre_model_path: Path = tmp_dir / self.name / self.key / f"pre_model.{model_suffix}"
        self.model_path: Path = tmp_dir / self.name / self.key / f"model.{model_suffix}"
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.params: dict = params
        self.initialize()

    def get_model_data(self):
        with open(self.model_path, "rb") as f:
            return f.read()

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def train(self, callback=None):
        pass

    async def receive_model(self, request: Request):
        logger.debug("Receiving model from %s to %s", request.remote, self.pre_model_path)
        with open(self.pre_model_path, "w+") as f:
            f.write("")
        async with aiofiles.open(self.pre_model_path, "ba+") as f:
            async for data in request.content.iter_chunked(10240):
                await f.write(data)
    
