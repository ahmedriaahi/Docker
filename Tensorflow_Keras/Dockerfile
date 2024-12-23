# Start from the TensorFlow official image
FROM tensorflow/tensorflow:latest-jupyter

# Set a working directory
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt /app/requirements.txt

# Copy your Python script into the container
COPY tensorflow_keras.py /app/tensorflow_keras.py

# Update system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Expose Jupyter Notebook port
EXPOSE 8888

# Start Jupyter Notebook server by default
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--Notebook.app=''", "--port=8888", "--no-browser", "--allow-root"]

