from typing import Optional, Dict, Any, List

import json
# import keras.models
# import numpy
# from keras.models import Sequential
# from keras.layers import *
# import tensorflow as tf


def initialize(model_weights: str, **kwargs) -> Dict[str, Any]:
    # clusters = kwargs.get("clusters", 4)
    # columns = kwargs.get("columns", [])
    centroids = kwargs.get("centroids", [])

    json_object = json.dumps(centroids)
    
    with open(model_weights, "w") as outfile:
        outfile.write(json_object)

    return kwargs


def get_weights(file: str):
    with open(file, 'r') as openfile:
        json_object = json.load(openfile)
    return json_object


def average_weights(model_weights: str, full_model: str, files: List[str]) -> None:
    centroids = []
    for file in files:
        with open(file, 'r') as openfile:
            json_object = json.load(openfile)
            centroids.append(json_object)

    # average centroids
    average_centroids = []

    json_object = json.dumps(average_centroids)
    
    with open(model_weights, "w") as outfile:
        outfile.write(json_object)
    with open(full_model, "w") as outfile:
        outfile.write(json_object)

