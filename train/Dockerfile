

#FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
#pytorch base image
FROM nvcr.io/nvidia/pytorch:23.06-py3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

# Set the working directory
WORKDIR /app

# Copy the Jupyter notebook and any additional files
COPY your_notebook.ipynb .
COPY .scripts/train.py scripts/.
COPY train_requirements.txt .
COPY .env .

# Install Python dependencies
RUN pip3 install jupyter
RUN pip3 install -r train_requirements.txt

# Expose the Jupyter notebook port
EXPOSE 8888

# Command to run Jupyter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]