FROM python:3.10-slim

RUN apt-get update && \
    apt-get clean

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

WORKDIR /app
COPY . /app
