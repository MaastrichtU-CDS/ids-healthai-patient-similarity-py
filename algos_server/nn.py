import json
import logging
from typing import Any, Dict, Optional

import keras.models
import tensorflow as tf
import numpy
from keras.layers import *
from keras.models import Sequential

from .algo import FederatedServerAlgo

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

class NNFederatedServerAlgo(FederatedServerAlgo):
    def __init__(self, params: Dict[str, Any]):
        # Doc: "'.h5' suffix causes weights to be saved in HDF5 format."
        # https://www.tensorflow.org/api_docs/python/tf/keras/Model#save_weights
        # Otherwise the path itself won't be created, instead: index and .data-00000-of-00001 files
        super().__init__(name="nn", params=params, model_suffix="h5")
        self.model: Optional[keras.Model] = None

    def initialize(self) -> None:
        model_input = self.params.get("model", None)
        if model_input is None:
            self.model = Sequential(
                [
                    Convolution2D(16, (5, 5), padding="valid", input_shape=(28, 28, 1)),
                    BatchNormalization(),
                    Activation("relu"),
                    MaxPooling2D(2, 2),
                    Convolution2D(16, (5, 5)),
                    BatchNormalization(),
                    Activation("relu"),
                    MaxPooling2D(2, 2),
                    Flatten(),
                    Dense(512, activation="relu"),
                    Dense(units=10, activation="softmax"),
                ]
            )
            self.params["model"] = self.model.get_config()
        else:
            try:
                self.model = tf.keras.models.model_from_json(model_input)
            except TypeError:
                model_input = json.dumps(model_input)
                self.model = tf.keras.models.model_from_json(model_input)

        optimizer_input = self.params.get("optimizer", None)
        if optimizer_input is None:
            optimizer = tf.keras.optimizers.Adam(0.001)
            self.params["optimizer"] = tf.keras.optimizers.serialize(optimizer)
        else:
            optimizer = tf.keras.optimizers.deserialize(optimizer_input)

        loss = self.params.setdefault("loss", "categorical_crossentropy")
        metrics = self.params.setdefault("metrics", ["accuracy"])

        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        # I guess here we are saving some initial random weights?
        logger.info("Saving some initial random (?) weights")
        self.model.save_weights(self.model_aggregated_path)

    def get_weights(self, file: str):
        self.model.load_weights(file)
        return self.model.get_weights()

    def aggregate(self, current_round):
        # take average between all partial models from all workers
        logger.info("Averaging %s into %s", self.get_partial_files(current_round), self.model_aggregated_path)
        weights = [self.get_weights(file) for file in self.get_partial_files(current_round)]
        array_weights = [numpy.array([w]) for w in weights]
        logger.info("Loaded weights")
        new_weights = list()
        for weights_list_tuple in zip(*array_weights):
            new_weights.append(
                numpy.array(
                    [
                        numpy.array(weights_).mean(axis=0)
                        for weights_ in zip(*weights_list_tuple)
                    ]
                )
            )
        logger.info("Calculated average weights")
        self.model.set_weights(new_weights[0])
        logger.info("Saving average weights")
        self.model.save_weights(self.model_aggregated_path)
        # TODO: save full model & offer via common_model
        #model.save(full_model)

        # for nn, we don't have a finish criteria, server handler will terminate after set number of rounds
        return False
