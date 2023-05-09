import logging
from typing import Optional, Any, Dict, Callable, Tuple

import json
import keras.models
import keras.callbacks
import pandas as pd

from numpy import ndarray
import tensorflow as tf
import numpy as np

logger = logging.getLogger("helper_worker_logger")

data: Optional[ndarray] = None
labels: Optional[ndarray] = None
model: Optional[keras.Model] = None
unique_labels: Optional[int] = None
array_features_dict: dict = {}


def initialize(**kwargs):
    global data
    global labels
    global model
    global unique_labels
    global array_features_dict

    key = kwargs.get("key")
    model_input = kwargs.get("model")
    optimizer_input = kwargs.get("optimizer")
    label_column = kwargs.get("label_column", "label")

    print(f"Reading data from {key}.csv")
    data = pd.read_csv(f"{key}.csv")
    print(f"Data shape: {data.shape}")

    features_dict = data.copy()
    for name, column in features_dict.items():
        dtype = column.dtype
        if dtype == object:
            features_dict[name] = features_dict[name].fillna("")

    labels = data[label_column]
    labels = pd.get_dummies(labels)

    array_features_dict = {
        name: np.array(value) for name, value in features_dict.items()
    }

    optimizer = tf.keras.optimizers.deserialize(optimizer_input)
    loss = kwargs.get("loss", "categorical_crossentropy")
    metrics = kwargs.get("metrics", ["accuracy"])

    if model_input:
        try:
            model = tf.keras.models.model_from_json(model_input)
        except TypeError:
            model_input = json.dumps(model_input)
            model = tf.keras.models.model_from_json(model_input)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)


def train(
    key: str,
    starting_weights: str,
    epochs: int,
    callback: Callable[[Dict[str, Any]], None],
    batch_size=8,
    validation_split=0.1,
) -> Tuple[str, str, str, str]:
    global data
    global labels
    global model
    global features_dict
    print(
        f"Start training ({key}, {starting_weights}, {epochs}, {batch_size}, {validation_split})"
    )
    if starting_weights is not None:
        print("Loading weights")
        model.load_weights(starting_weights)
    print("Setup callback")

    class KerasCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f"End epoch {epoch}; {logs}")
            callback({**logs, "epoch": epoch})

    print("Start fitting")
    print(data)
    print(features_dict)
    try:
        model.fit(
            features_dict,
            labels,
            verbose=2,
            callbacks=[KerasCallback()],
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
        )
        print("Finished fitting")

        model.save_weights(f"weights.{key}.h5")
    except Exception as e:
        logger.exception(e)
    return True
