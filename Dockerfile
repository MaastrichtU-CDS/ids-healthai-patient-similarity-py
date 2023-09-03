FROM tensorflow/tensorflow:2.10.0

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

WORKDIR /app
COPY . /app
