# Template of Dockerfile, this is python 3.7 on a very small debian distro
FROM python:3.9-slim-buster

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Debian installation of necessary packages
RUN apt-get update \
  && apt-get install --reinstall build-essential -y

# RUN add-apt-repository ppa:graphics-drivers/ppa \ 
# && apt update \
# && apt install nvidia-390 \
# && apt install nvidia-cuda-toolkit 

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies. Mandatory: Flask, gunicorn, flask_cors
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# This expects a main python script in the same directory level as this Dockerfile called app.py
CMD exec gunicorn --bind :$PORT --workers 4 --threads 8 --timeout 0 app:app
