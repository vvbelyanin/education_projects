# part2_dvc/scripts/fit.py

import pandas as pd
import yaml
import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from catboost import CatBoostRegressor

def fit_model():
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd) 
	
    data = pd.read_csv('data/initial_data.csv')
    cat_features= ['is_apartment', 'has_elevator', 'rooms', 'building_type_int']
    target = ['price']
    
    X_train, X_test, y_train, y_test = train_test_split(data.drop(target, axis=1), 
                                                        data[target], 
                                                        random_state=params['random_state']) 
    
    preprocessor = ColumnTransformer([
         ('one_hot_drop', OneHotEncoder(
              drop=params['one_hot_drop'], 
              sparse_output=False), 
              cat_features)
              ],
        remainder='passthrough',
        verbose_feature_names_out=False
        )
    model = CatBoostRegressor(learning_rate=params['learning_rate'], 
                              iterations=params['iterations'], 
                              depth=params['depth'], 
                              loss_function=params['loss_function'],
                              random_seed=params['random_state'], 
                              verbose=0)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)
	
    os.makedirs('models', exist_ok=True)
    with open('models/fitted_model.pkl', 'wb') as fd:
        joblib.dump(pipeline, fd) 
    
    os.makedirs('data', exist_ok=True)
    X_train.to_csv('data/x_train.csv', index=None)
    y_train.to_csv('data/y_train.csv', index=None)
    X_test.to_csv('data/x_test.csv', index=None)
    y_test.to_csv('data/y_test.csv', index=None)

if __name__ == '__main__':
	fit_model()