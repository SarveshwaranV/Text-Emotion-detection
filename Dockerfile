# Use a compatible Python runtime as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    && apt-get clean

# Copy the requirements file to the working directory
COPY requirements.txt .

# Upgrade pip to the latest version and install dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Define the environment variable for Flask
ENV FLASK_APP=app.py

# Command to run the application
CMD ["python", "app.py"]
