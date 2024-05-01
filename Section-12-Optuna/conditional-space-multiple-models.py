import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

import optuna

# Load the breast cancer dataset
breast_cancer_X, breast_cancer_y = load_breast_cancer(return_X_y=True)
X = pd.DataFrame(breast_cancer_X)
y = pd.Series(breast_cancer_y).map({0: 1, 1: 0})  # Invert the target variable

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
    classifier_name = trial.suggest_categorical('classifier', ['LogisticRegression', 'RandomForest', 'GradientBoosting'])

    if classifier_name == 'RandomForest':
        rf_n_estimators = trial.suggest_int('rf_n_estimators', 100, 1000)
        rf_max_depth = trial.suggest_int('rf_max_depth', 1, 4)
        rf_min_samples_split = trial.suggest_float('rf_min_samples_split', 0.01, 1.0)
        rf_criterion = trial.suggest_categorical('rf_criterion', ['gini', 'entropy'])
        classifier_obj = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            min_samples_split=rf_min_samples_split,
            criterion=rf_criterion
        )
    elif classifier_name == 'GradientBoosting':
        gb_n_estimators = trial.suggest_int('gb_n_estimators', 100, 1000)
        gb_criteria = trial.suggest_categorical('gb_criteria', ['squared_error', 'friedman_mse'])
        gb_learning_rate = trial.suggest_float('gb_learning_rate', 0.01, 0.1)
        gb_max_depth = trial.suggest_int('gb_max_depth', 1, 4)
        gb_min_samples_split = trial.suggest_float('gb_min_samples_split', 0.01, 1.0)
        
        classifier_obj = GradientBoostingClassifier(
            n_estimators=gb_n_estimators,
            criterion=gb_criteria,
            learning_rate=gb_learning_rate,
            max_depth=gb_max_depth,
            min_samples_split=gb_min_samples_split
        )
    else:  # LogisticRegression
        lr_C = trial.suggest_float('lr_C', 1e-3, 10)
        lr_penalty = trial.suggest_categorical('lr_penalty', ['l1', 'l2'])
        lr_solver = 'saga' # trial.suggest_categorical('lr_solver', ['saga', 'liblinear'])
        classifier_obj = LogisticRegression(
            C=lr_C,
            penalty=lr_penalty,
            solver=lr_solver  # 'saga', 'liblinear' solver supports both 'l1' and 'l2' penalties
        )

    cv = 3
    score = cross_val_score(classifier_obj, X_train, y_train, cv=cv)
    accuracy = score.mean()

    return accuracy

# Create the study object
def run_optuna_study(sampler_name, sampler):
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=30)
    
    print(f"{sampler_name} - Best params: {study.best_params}")
    print(f"{sampler_name} - Best value: {study.best_value}")
    print(f"{sampler_name} - Trials dataframe:\n{study.trials_dataframe()}")
    results = study.trials_dataframe()
    #results['params_classifier'].value_counts().plot(kind='bar')
    print(results['params_classifier'].value_counts())
    #results.groupby('params_classifier')['value'].agg(['mean', 'std'])
    print(results.groupby('params_classifier')['value'].agg(['mean', 'std']))

    results['value'].sort_values().reset_index(drop=True).plot()
    plt.title(f"{sampler_name} - Convergence Plot")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    #Text(0, 0.5, 'Accuracy')

# Test different Samplers
run_optuna_study("CamEsSampler", optuna.samplers.TPESampler())
