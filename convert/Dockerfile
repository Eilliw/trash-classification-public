
FROM nvcr.io/nvidia/cuda:12.3.0-devel-ubuntu20.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

# Set the working directory
WORKDIR /app

# Copy the Jupyter notebook and any additional files

# Install Python dependencies
RUN pip3 install jupyter opencv-python

# Expose the Jupyter notebook port
EXPOSE 8888