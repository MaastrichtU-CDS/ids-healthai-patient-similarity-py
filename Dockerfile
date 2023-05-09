FROM tensorflow/tensorflow:2.10.0
# FROM tensorflow/tensorflow:2.7.0-gpu

COPY ./requirements.txt /requirements.txt

RUN pip install -r /requirements.txt

WORKDIR /app

COPY . /app