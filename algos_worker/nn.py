import json
import logging
from typing import Any, Callable, Dict, Optional

import keras.callbacks
import keras.models
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical
from numpy import ndarray

from .algo import FederatedWorkerAlgo

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

class NNFederatedWorkerAlgo(FederatedWorkerAlgo):
    def __init__(self, params: dict = {}):
        self.data: Optional[ndarray] = None #
        self.labels: Optional[ndarray] = None
        self.model: Optional[keras.Model] = None
        self.unique_labels: Optional[int] = None
        self.array_features_dict: dict = {}
        self.input_type: Optional[str] = None
        super().__init__(name="nn", params=params, model_suffix="h5")

    def initialize(self):
        self.input_type = self.params.get("input_type", "array")
        logger.info("Reading data from %s.csv", self.key)
        self.data = pd.read_csv(f"{self.key}.csv")
        logger.info("Data shape: %s", self.data.shape)
        label_column = self.params.get("label_column", "label")
        self.labels = self.data[label_column]
        self.batch_size = self.params.get("batch_size", 8)
        self.validation_split = self.params.get("validation_split", 0.1)
        self.epochs = self.params.get("epochs", 3)

        model_input = self.params.get("model", {
            "class_name": "Sequential",
            "config": {
                "name": "sequential",
                "layers": [
                {
                    "class_name": "InputLayer",
                    "config": {
                    "batch_input_shape": [
                        None,
                        784,
                        1
                    ],
                    "dtype": "float32",
                    "name": "reshape_input"
                    }
                },
                {
                    "class_name": "Reshape",
                    "config": {
                    "name": "reshape",
                    "dtype": "float32",
                    "batch_input_shape": [
                        None,
                        784,
                        1
                    ],
                    "target_shape": [
                        28,
                        28,
                        1
                    ]
                    }
                },
                {
                    "class_name": "Rescaling",
                    "config": {
                    "scale": 0.003921568627,
                    "offset": 0
                    }
                },
                {
                    "class_name": "Conv2D",
                    "config": {
                    "name": "conv2d",
                    "trainable": True,
                    "batch_input_shape": [
                        None,
                        28,
                        28,
                        1
                    ],
                    "dtype": "float32",
                    "filters": 16,
                    "kernel_size": [
                        5,
                        5
                    ],
                    "activation": "linear"
                    }
                },
                {
                    "class_name": "BatchNormalization",
                    "config": {
                    "name": "batch_normalization",
                    "dtype": "float32",
                    "axis": [
                        3
                    ]
                    }
                },
                {
                    "class_name": "Activation",
                    "config": {
                    "name": "activation",
                    "dtype": "float32",
                    "activation": "relu"
                    }
                },
                {
                    "class_name": "MaxPooling2D",
                    "config": {
                    "name": "max_pooling2d",
                    "dtype": "float32"
                    }
                },
                {
                    "class_name": "Conv2D",
                    "config": {
                    "name": "conv2d_1",
                    "dtype": "float32",
                    "filters": 16,
                    "kernel_size": [
                        5,
                        5
                    ],
                    "activation": "linear"
                    }
                },
                {
                    "class_name": "BatchNormalization",
                    "config": {
                    "name": "batch_normalization_1",
                    "trainable": True,
                    "dtype": "float32",
                    "axis": [
                        3
                    ]
                    }
                },
                {
                    "class_name": "Activation",
                    "config": {
                    "name": "activation_1",
                    "dtype": "float32",
                    "activation": "relu"
                    }
                },
                {
                    "class_name": "MaxPooling2D",
                    "config": {
                    "name": "max_pooling2d_1",
                    "dtype": "float32"
                    }
                },
                {
                    "class_name": "Flatten",
                    "config": {
                    "name": "flatten",
                    "dtype": "float32"
                    }
                },
                {
                    "class_name": "Dense",
                    "config": {
                    "name": "dense",
                    "dtype": "float32",
                    "units": 512,
                    "activation": "relu"
                    }
                },
                {
                    "class_name": "Dense",
                    "config": {
                    "name": "dense_1",
                    "dtype": "float32",
                    "units": 10,
                    "activation": "softmax"
                    }
                }
                ]
            },
            "keras_version": "2.7.0",
            "backend": "tensorflow"
        })
        optimizer_input = self.params.get("optimizer", {
            "class_name": "Adam",
            "config": {
                "name": "Adam",
                "learning_rate": 0.00001,
                "decay": 0,
                "beta_1": 0.9,
                "beta_2": 0.999,
                "epsilon": 1e-7,
                "amsgrad": False
            }
        })

        optimizer = tf.keras.optimizers.deserialize(optimizer_input)
        loss = self.params.get("loss", "categorical_crossentropy")
        metrics = self.params.get("metrics", ["accuracy"])

        if self.input_type == "dict":
            features_dict = self.data.copy()
            for name, column in features_dict.items():
                dtype = column.dtype
                if dtype == object:
                    features_dict[name] = features_dict[name].fillna("")


            self.array_features_dict = {
                name: np.array(value) for name, value in features_dict.items()
            }
        else:
            num_classes = self.params.get("num_classes", 0)
            self.unique_labels = len(np.unique(self.labels))
            num_classes = self.unique_labels if num_classes == 0 else num_classes
            if loss == "categorical_crossentropy":
                self.labels = to_categorical(self.labels, num_classes=num_classes)

            self.data = self.data.drop([label_column], axis=1)

        if model_input:
            try:
                self.model = tf.keras.models.model_from_json(model_input)
            except TypeError:
                model_input = json.dumps(model_input)
                self.model = tf.keras.models.model_from_json(model_input)
        logger.info("Compiling model")
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)


    def train(self, callback: Callable[[Dict[str, Any]], None] = None) -> bool:
        logger.info("Starting training (%s, %s, %s, %s, %s)", self.key, self.pre_model_path, self.epochs, self.batch_size, self.validation_split)

        logger.info("Loading provided starting weights")
        #self.model.load_weights(self.pre_model_path)

        logger.info("Setting up callback")
        class KerasCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                print(f"End epoch {epoch}; {logs}")
                callback({**logs, "epoch": epoch})

        logger.info("Starting fitting")
        if self.input_type == "dict":
            print(self.array_features_dict.keys())
        else:
            print(self.data.shape)
        try:
            self.model.fit(
                self.array_features_dict if self.input_type == "dict" else self.data,
                self.labels,
                verbose=2,
                callbacks=[KerasCallback()],
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=self.validation_split,
            )
            logger.info("Finished fitting")

            logger.info("Saving model")
            self.model.save_weights(self.model_path)
        except Exception as e:
            logger.exception(e)
        return True
