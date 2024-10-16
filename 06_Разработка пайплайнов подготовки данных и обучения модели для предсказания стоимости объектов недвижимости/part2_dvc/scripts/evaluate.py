# part2_dvc/scripts/evaluate.py

import pandas as pd
from sklearn.model_selection import KFold, cross_validate
import joblib
import json
import yaml
import os

from sklearn.metrics import root_mean_squared_error, mean_absolute_error

def evaluate_model():
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd) 

    X_train = pd.read_csv('data/x_train.csv')
    y_train = pd.read_csv('data/y_train.csv')
    X_test = pd.read_csv('data/x_test.csv')
    y_test = pd.read_csv('data/y_test.csv')

    with open('models/fitted_model.pkl', 'rb') as fd:
        model = joblib.load(fd)

    cv_strategy = KFold(n_splits=params['n_splits'], random_state=params['random_state'], shuffle=True)
    cv_res = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv_strategy,
        n_jobs=params['n_jobs'],
        scoring=params['metrics'],
        verbose=2
        )
    for key, value in cv_res.items():
        cv_res[key] = round(value.mean(), 4) if value.mean()>0 else -round(value.mean(), 4)
	
    y_pred = model.predict(X_test)
    cv_res['final_mae'] = round(mean_absolute_error(y_test, y_pred), 4)
    cv_res['final_rmse'] = round(root_mean_squared_error(y_test, y_pred), 4)
    
    os.makedirs('cv_results', exist_ok=True)
    with open("cv_results/cv_res.json", "w") as outfile:
        json.dump(cv_res, outfile)
    
    for key, value in cv_res.items():
        print(f'{key}: {value}')

if __name__ == '__main__':
	evaluate_model()