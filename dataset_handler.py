import logging
import os
import sys
from datetime import datetime
from io import BytesIO
from typing import List, Dict, Any

import pandas as pd
import requests
from aiohttp import web

logger = logging.getLogger("dataset_handler")
data_app_url = os.environ.get('DATA_APP_URL', 'https://httpbin.org/anything')


class DataSetHandler:
    _datasets: List[Dict[str, Any]] = []

    async def delete_dataset(self, request: web.Request) -> web.Response:
        key = request.match_info.get('key')
        logger.info("deleting dataset with key {}".format(key))
        if os.path.exists(f"{key}.csv"):
            os.remove(f"{key}.csv")
        self._datasets = [dataset for dataset in self._datasets if not dataset['name'] == key]
        response = requests.delete(f'{data_app_url}/datasets/{key}')
        if response.status_code > 299:
            logger.error(f'Error in deleting dataset metadata: {response.status_code}')
        else:
            logger.info('Deleted dataset metadata successfully')
        return web.Response()

    async def list_datasets(self, request: web.Request) -> web.Response:
        return web.json_response(self._datasets)

    async def add_dataset(self, request: web.Request) -> web.Response:
        """returns the metadata of the newly added dataset.
        Stores the dataset on the filesystem of the container
        """
        key = request.match_info.get('key')
        logger.info('received dataset with key {}.'.format(key))
        if key in self._datasets:
            raise web.HTTPConflict(text="Dataset with key %s already exists" % key)

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
        response = requests.post(f'{data_app_url}/datasets/{key}', json=response_dict)
        if response.status_code > 299:
            logger.error(f'Error in sharing dataset metadata: {response.status_code}')
        else:
            logger.info('Shared dataset metadata successfully')
        return web.json_response(response_dict)


if __name__ == "__main__":
    # Logging setup
    FORMAT = "%(asctime)s | %(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger("worker")
    logger.setLevel(level=logging.DEBUG)
