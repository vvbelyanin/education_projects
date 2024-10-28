"""
Filename: services/models/load_model.py

This script loads a pre-trained model from an S3 bucket, saves it locally, and
generates a random prediction using the loaded model.

1. Loads environment variables for S3 and AWS credentials.
2. Downloads the model file from the specified S3 bucket.
3. Loads the model using joblib.
4. Generates a random set of model parameters and makes a prediction.

Key Components:
- MODEL_NAME: Path to save the downloaded model file.
- ENV_VARS: List of required environment variables for S3 and AWS credentials.
"""

import os
import sys
from dotenv import load_dotenv
from geopy.distance import geodesic
import pandas as pd
import boto3
import joblib
from random import randint, uniform
from services.app import fastapi_handler

# Constants
MODEL_NAME = 'services/models/loaded_model.pkl'
ENV_VARS = [
    'S3_BUCKET_NAME',
    'AWS_ACCESS_KEY_ID',
    'AWS_SECRET_ACCESS_KEY',
    'MLFLOW_S3_ENDPOINT_URL',
    'MODEL_FILE_KEY'
]

# Load environment variables from .env file
load_dotenv(dotenv_path='/.env')

# Check if all required environment variables are defined
if not all(_ in os.environ for _ in ENV_VARS):
    print("Cannot load model: environment variables are not defined, check .env file.")
    sys.exit()

# Create an S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    endpoint_url=os.environ['MLFLOW_S3_ENDPOINT_URL']
)

# Download the model file from the S3 bucket
s3_client.download_file(os.getenv('S3_BUCKET_NAME'), os.getenv('MODEL_FILE_KEY'), MODEL_NAME)

# Load the model using joblib
model = joblib.load(MODEL_NAME)

print(f"Model {model.__class__.__module__}.{model.__class__.__name__}."
      f"{model._final_estimator} loaded successfully to: {MODEL_NAME}")

# Generate random model parameters and make a prediction
params = fastapi_handler.gen_random_data()

print(f"Random prediction: {model.predict(pd.DataFrame(params, index=[0]))}")
