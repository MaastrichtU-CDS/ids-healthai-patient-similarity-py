from typing import Optional, Dict, Any, List

import json
import keras.models
import numpy
from keras.models import Sequential
from keras.layers import *
import tensorflow as tf

model: Optional[keras.Model] = None


def initialize(model_weights: str, **kwargs) -> Dict[str, Any]:
    global model
    model_input = kwargs.get("model", None)
    if model_input is None:
        model = Sequential(
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
        kwargs["model"] = model.get_config()
    else:
        try:
            model = tf.keras.models.model_from_json(model_input)
        except TypeError:
            model_input = json.dumps(model_input)
            model = tf.keras.models.model_from_json(model_input)

    optimizer_input = kwargs.get("optimizer", None)
    if optimizer_input is None:
        optimizer = tf.keras.optimizers.Adam(0.001)
        kwargs["optimizer"] = tf.keras.optimizers.serialize(optimizer)
    else:
        optimizer = tf.keras.optimizers.deserialize(optimizer_input)

    loss = kwargs.setdefault("loss", "categorical_crossentropy")
    metrics = kwargs.setdefault("metrics", ["accuracy"])

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.save_weights(model_weights)
    return kwargs


def get_weights(file: str):
    model.load_weights(file)
    return model.get_weights()


def average_weights(model_weights: str, full_model: str, files: List[str]) -> None:
    print(f"Averaging {files} into {model_weights}")
    weights = [get_weights(file) for file in files]
    array_weights = [numpy.array([w]) for w in weights]
    print("Loaded weights")

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

    print("Calculated average weights")
    model.set_weights(new_weights[0])
    print("Saving average weights")
    model.save_weights(model_weights)
    model.save(full_model)
