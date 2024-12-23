FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Upgrade pip and install the Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script into the container
COPY cnn_tf.py .

# Expose the port for Jupyter
EXPOSE 8888

# Set the default command to run Jupyter
CMD ["jupyter", "notebook", "--Notebook.app=''" "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

