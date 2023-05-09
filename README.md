# Tensorflow backend

This repository contains the logic for the Federated Learning implementation through IDS. It is an implementation based on Tensorflow 2.11, and can be used to train Keras models through IDS communication. The setup of this data app is that it can receive a model and its parameters from a User Interface. This means that this component is setup to be as generic as possible, requiring no changes to the code to deploy your own use case. Therefore, the preprocessing of the data either needs to happen beforehand and saved to a CSV file, or you should include the preprocessing in the Layers of the Keras model.

The code in this repository needs the [Federated Learning Data App](https://gitlab.com/tno-tsg/data-apps/federated-learning) combined with a [TSG Core Container](https://gitlab.com/tno-tsg/core-container) to work. 

## Deployment
This Data App should be deployed via the [TSG Connector Helm Chart](https://gitlab.com/tno-tsg/helm-charts/connector), providing the following configuration for a worker:
```yaml 
containers:
  # ... Federated Learning data app config here
  - type: helper
    image: docker.nexus.dataspac.es/federated-learning/tf-backend:1.0.0 # Image of this repository
    name: tf-backend
    command: ["python3"]
    args: ["federated_learning.py"]
    tty: true
    environment:
      - name: DATA_APP_URL
        value: http://{{ template "tsg-connector.fullname" . }}-federated-learning-http:8080
    services:
      - port: 8080
        name: http
        ingress:
          path: /tf/(.*)
          rewriteTarget: /$1
          clusterIssuer: letsencrypt
```

Change the `federated_learning.py` to `federated_learning_server.py` for the researcher configuration.

## Endpoints

The Federated Learning Data App contains the following endpoints:

### Researcher

| Endpoint | Method | Request | Response | Description |
|---|---|---|---|---|
| `initialize` | `POST` | {"title": string,"rounds": int,"epochs": int,"batch_size": int ,"validation_split": float ,"normalization": string,"shape":[int],"label_column": string,"loss": string,"metrics":[string],"model": {JSON structure of Keras model},"optimizer":{JSON structure of Keras Optimizer},"workers":[string],"key": string} | - | Initialize the Federated Learning process with some necessary parameters. |
| `train` | `POST` | - | - | Start the training of the model. |
| `model` | `POST` | h5 model | - | Endpoint for the researcher to share the model to. If all the workers have shared their model, an average performance metric is calculated. |
| `status` | `GET` | - | List of metrics, as given in the 'initialize' call e.g. [[{"loss": float,"val_loss": float,"root_mean_squared_error": float,"round": int,"epoch": int,"worker": string,"val_root_mean_squared_error":float ,"timestamp": int}]] | Get the status of the federated learning, a list of results is published, each connectors produces its own list. The lists per connector contain one entry per epoch. |
| `model` | `GET` | - | Full model in h5 format | Download the model in h5 format. |

### Worker

| Endpoint | Method | Request | Response | Description |
|---|---|---|---|---|
| `initialize` | `POST` | {"title": string,"rounds": int,"epochs": int,"batch_size": int ,"validation_split": float ,"normalization": string,"shape":[int],"label_column": string,"loss": string,"metrics":[string],"model": {JSON structure of Keras model},"optimizer":{JSON structure of Keras Optimizer},"workers":[string],"key": string} | - | Initialize the Federated Learning process with some necessary parameters for the specific worker. |
| `model` | `POST` | - | - | Start the training of the model for the specific worker. |
| `finish` | `POST` | - | - | Finish federated learning |
| `status` | `GET` | - | {"state": NONE|INITIALIZED|WAITING|TRAINING|FINISHED, "status": []} | Get the status of the Federated learning. |
| `/datasets/{key}` | `POST` | {key: string, dataset: string} | - | Stores the dataset on the filesystem of the container. |
| `/datasets/{key}` | `DELETE` | key: string | - | Delete the file from the filesystem of the container. |
| `/datasets` | `GET` | - | [{'name': string, 'shape': tuple[int], 'byteSize': int, 'mediaType': 'text/csv', 'creationDate': string}, ] | List all datasets of this worker.