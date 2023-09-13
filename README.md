# :warning: :warning:

This repository ended up being used as the basis for the algorithms both for server and worker for the IDS solution for the HealthAI demo.
It's trully a proof-of-concept. Implements kmeans (v6/Aiara), statistics (v6/Aiara), and neuralnet (TNO/Maarten).

However it does not handle multiple users, multiple tasks at the same time, authentication, etc. 

# Patient similarity backend
This repository contains the logic for the Federated Learning implementation 
through IDS for patient similarity with TNM data. Neural net training should also be possible :construction:.
Different algorithm to collect statistics should be feasible to implement. This is a proof-of-concept.

The code in this repository needs the [Federated Learning Data App](https://gitlab.com/tno-tsg/data-apps/federated-learning) 
combined with a [TSG Core Container](https://gitlab.com/tno-tsg/core-container) to work. 

## Deployment

This Data App should be deployed via the [TSG Connector Helm Chart](https://gitlab.com/tno-tsg/helm-charts/connector), 
providing the following configuration for a worker:

```yaml 
containers:
  # ... Federated Learning data app config here
  - type: helper
    image: ghcr.io/maastrichtu-cds/ids-healthai-patient-similarity-py:demo 
    name: patient-similarity-backend
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

Change the `federated_learning.py` and `federated_learning_server.py` for the 
researcher configuration.

## Acknowledgments

This project was financially supported by the
[AiNed foundation](https://ained.nl/over-ained/).
