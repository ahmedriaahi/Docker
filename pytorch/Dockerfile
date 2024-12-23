# Use an official PyTorch runtime as the base image
FROM pytorch/pytorch:latest

# Set a working directory
WORKDIR /app

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install required packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-setuptools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the working directory
COPY requirements.txt ./requirements.txt

# Copy the Python script into the working directory
COPY pytorch.py ./pytorch.py

# Install the required Python packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Install JupyterLab
RUN pip3 install --no-cache-dir jupyterlab

# Expose the Jupyter Lab port
EXPOSE 8888

# Start Jupyter Lab on the exposed port
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--Notebook.app=''", --port=8888", "--no-browser", "--allow-root"]

