"""
services/models/create_model.py

This script trains a CatBoost regression model using data from a PostgreSQL database
and saves the fitted model to a file.

1. Loads environment variables for database connection.
2. Connects to the PostgreSQL database and retrieves data from the specified table.
3. Prepares the data for training by separating features and target variable.
4. Trains a CatBoost regression model.
5. Saves the trained model to a pickle file.

Key Components:
- TABLE_NAME: Name of the table in the database containing the training data.
- RANDOM_STATE: Random state for model reproducibility.
- MODEL_NAME: Path to save the trained model.
- ENV_VARS: List of required environment variables for database connection.
"""

import os
import sys
import pickle
import pandas as pd
from sqlalchemy import create_engine
from catboost import CatBoostRegressor
from dotenv import load_dotenv

# Constants
TABLE_NAME = 'clean_flats'
RANDOM_STATE = 42
MODEL_NAME = 'services/models/fitted_model.pkl'
ENV_VARS = [
    'DB_DESTINATION_USER',
    'DB_DESTINATION_PASSWORD',
    'DB_DESTINATION_HOST',
    'DB_DESTINATION_PORT',
    'DB_DESTINATION_NAME'
]

# Load environment variables from .env file
load_dotenv(dotenv_path='/.env')

# Check if all required environment variables are defined
if not all(_ in os.environ for _ in ENV_VARS):
    print("Cannot create model: environment variables are not defined, check .env file.")
    sys.exit()

# Create a connection to the PostgreSQL database
con = create_engine(
    f"postgresql://{os.getenv('DB_DESTINATION_USER')}:"
    f"{os.getenv('DB_DESTINATION_PASSWORD')}@"
    f"{os.getenv('DB_DESTINATION_HOST')}:"
    f"{os.getenv('DB_DESTINATION_PORT')}/"
    f"{os.getenv('DB_DESTINATION_NAME')}"
)

# Load data from the database
df = pd.read_sql(sql=f'SELECT * FROM {TABLE_NAME}', con=con).drop(['id', 'flat_id'], axis=1)
X, y = df.drop('price', axis=1), df['price']

# Train the CatBoost regression model
model = CatBoostRegressor(verbose=0, random_state=RANDOM_STATE)
model.fit(X, y)

# Save the trained model to a pickle file
with open(MODEL_NAME, 'wb') as model_file:
    pickle.dump(model, model_file)

print(f'Fitted model saved to {MODEL_NAME}')
