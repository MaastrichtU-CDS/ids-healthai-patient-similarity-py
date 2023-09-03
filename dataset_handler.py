import logging
import os
import sys
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List

import pandas as pd
import requests
from aiohttp import web

# TODO: datasets are getting saved as .csv.csv ..

data_app_url = os.environ.get('DATA_APP_URL', 'http://127.0.11.1:8080')

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

class DataSetHandler:
    _datasets: List[Dict[str, Any]] = []

    async def delete_dataset(self, request: web.Request) -> web.Response:
        key = request.match_info.get('key')
        logger.info("Deleting dataset with key %s", key)
        if os.path.exists(f"{key}.csv"):
            os.remove(f"{key}.csv")
        self._datasets = [dataset for dataset in self._datasets if not dataset['name'] == key]
        response = requests.delete(f'{data_app_url}/datasets/{key}', timeout=20)
        if response.status_code > 299:
            logger.error("Error in deleting dataset metadata: %s", response.status_code)
        else:
            logger.info("Deleted dataset metadata successfully")
        return web.Response()

    async def list_datasets(self, request: web.Request) -> web.Response:
        logger.info("Listing datasets")
        return web.json_response(self._datasets)

    async def add_dataset(self, request: web.Request) -> web.Response:
        """returns the metadata of the newly added dataset.
        Stores the dataset on the filesystem of the container
        """
        print("Received a data set")
        key = request.match_info.get('key')
        logger.info("Received dataset with key %s", key)
        if key in self._datasets:
            raise web.HTTPConflict(text=f"Dataset with key {key} already exists")

        csv = await request.read()
        f = open(f"{key}.csv", "wb")
        f.write(csv)
        f.close()

        response_dict = {
            'name': key,
            'shape': pd.read_csv(BytesIO(csv)).shape,
            'byteSize': sys.getsizeof(csv),
            'mediaType': 'text/csv',
            'creationDate': datetime.now().isoformat()
        }
        self._datasets.append(response_dict)
        response = requests.post(f'{data_app_url}/datasets/{key}', json=response_dict, timeout=20)
        if response.status_code > 299:
            logger.error("Error in sharing dataset metadata: %s", response.status_code)
        else:
            logger.info('Shared dataset metadata successfully')
        return web.json_response(response_dict)


# TODO: if running as standalone?
if __name__ == "__main__":
    # Logging setup
    FORMAT = "%(asctime)s | %(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
