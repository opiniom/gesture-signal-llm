# Use the base image from Intel dlstreamer
FROM intel/dlstreamer:devel

USER root

# Set environment variables for display and user
ENV DISPLAY=$DISPLAY

# Install packages
RUN apt-get update -y && \
    apt-get install vim -y && \
    apt-get install git -y && \
    pip install scikit-learn && \
    python3 -m pip install --upgrade pip

# Download models (This is optional if you need it)
WORKDIR /opt/intel/dlstreamer/samples
COPY models.lst .
RUN ./download_models.sh

# Copy and install ActionAI
COPY . /home/dlstreamer/ActionAI
RUN pip install /home/dlstreamer/ActionAI

WORKDIR /home/dlstreamer
