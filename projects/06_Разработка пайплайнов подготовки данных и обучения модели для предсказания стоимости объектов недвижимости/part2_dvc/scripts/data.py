# part2_dvc/scripts/data.py

import pandas as pd
import os
import yaml
from sqlalchemy import create_engine
from dotenv import load_dotenv

def create_connection():

    load_dotenv()
    host = os.environ.get('DB_DESTINATION_HOST')
    port = os.environ.get('DB_DESTINATION_PORT')
    db = os.environ.get('DB_DESTINATION_NAME')
    username = os.environ.get('DB_DESTINATION_USER')
    password = os.environ.get('DB_DESTINATION_PASSWORD')
    
    conn = create_engine(f'postgresql://{username}:{password}@{host}:{port}/{db}', connect_args={'sslmode':'require'})

    return conn

def get_data():

    conn = create_connection()
    data = pd.read_sql('select * from clean_flats', conn).drop(['id', 'flat_id'], axis=1)
    conn.dispose()

    os.makedirs('data', exist_ok=True)
    data.to_csv('data/initial_data.csv', index=None)

if __name__ == '__main__':
    get_data()