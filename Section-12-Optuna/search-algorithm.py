import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

import optuna

# Load the breast cancer dataset
breast_cancer_X, breast_cancer_y = load_breast_cancer(return_X_y=True)
X = pd.DataFrame(breast_cancer_X)
y = pd.Series(breast_cancer_y)

# Check the data
print(X.head())
print(y.value_counts() / len(y))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Check the shapes
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Define the objective function
# This is the hyperparameter response space, the function we want to minimize
def objective(trial):
    rf_n_estimators = trial.suggest_int('rf_n_estimators', 100, 1000)
    rf_max_depth = trial.suggest_int('rf_max_depth', 1, 4)
    rf_min_samples_split = trial.suggest_float('rf_min_samples_split', 0.01, 1)
    rf_criterion = trial.suggest_categorical('rf_criterion', ['gini', 'entropy'])
    
    model = RandomForestClassifier(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        min_samples_split=rf_min_samples_split,
        criterion=rf_criterion,
        #random_state=0
    )
    cv = 3
    score = cross_val_score(model, X_train, y_train, cv=cv)
    accuracy = score.mean()
    
    return accuracy

# Create the study object
def run_optuna_study(sampler_name, sampler):
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=5)
    
    print(f"{sampler_name} - Best params: {study.best_params}")
    print(f"{sampler_name} - Best value: {study.best_value}")
    print(f"{sampler_name} - Trials dataframe:\n{study.trials_dataframe()}")

# Test different Samplers
run_optuna_study("RandomSampler", optuna.samplers.RandomSampler())
run_optuna_study("TPESampler", optuna.samplers.TPESampler())
run_optuna_study("CamEsSampler", optuna.samplers.CmaEsSampler())

# Create search_space for GridSampler which is special
search_space = {
    'rf_n_estimators': [100, 500, 1000],
    'rf_max_depth': [1, 2, 3],
    'rf_min_samples_split': [0.1, 1.0],
    'rf_criterion': ['gini', 'entropy']
}
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.GridSampler(search_space))
study.optimize(objective)
