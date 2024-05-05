## Clone Repository from https://github.com/solegalli/hyperparameter-optimization.git
```
## 1. Clone tahu
git clone https://github.com/solegalli/hyperparameter-optimization.git
## 2. Rename orignal remote as 'upstream' that free 'origin' for this respository remote URL
git remote rename origin upstream 
## 3. create repository at github account, eg, https://github.com/david-yyongning/hyperparameter-optimization-learn.git
## 4. Add `origin` point to new Github respository
git remote add origin https://github.com/david-yyongning/hyperparameter-optimization-learn.git
## 5. push only master to remote repository named "origin", '-u' set local master upstream branch to origin/master 
git push -u origin master
```

## Package Installation
```sh
pip install h5py # conda install h5py not working
pip install torch botorch gpytorch # for TrRBO algorithm
```
 
 ![PythonVersion](https://img.shields.io/badge/python-3.6%20|3.7%20|%203.8%20|%203.9-success)
[![License https://github.com/solegalli/hyperparameter-optimization/blob/master/LICENSE](https://img.shields.io/badge/license-BSD-success.svg)](https://github.com/solegalli/hyperparameter-optimization/blob/master/LICENSE)
[![Sponsorship https://www.trainindata.com/](https://img.shields.io/badge/Powered%20By-TrainInData-orange.svg)](https://www.trainindata.com/)

## Hyperparameter tuning for Machine Learning - Code Repository

[<img src="./course-banner.png">](https://www.trainindata.com/p/hyperparameter-optimization-for-machine-learning)

**Launched**: May, 2021

[<img src="./logo.png" width="248">](https://www.trainindata.com/p/hyperparameter-optimization-for-machine-learning)

## Links

- [Online Course](https://www.trainindata.com/p/hyperparameter-optimization-for-machine-learning)


## Table of Contents


1. **Cross-Validation**
	1. K-fold, LOOCV, LPOCV, Stratified CV
	2. Group CV and variants
	3. CV for time series
	4. Nested CV

2. **Basic Search Algorithms**
	1. Manual Search, Grid Search and Random Search

3. **Bayesian Optimization**
	1. with Gaussian Processes
	2. with Random Forests (SMAC) and GBMs
	3. with Parzen windows (Tree-structured Parzen Estimators or TPE)

4. **Multi-fidelity Optimization**
	1. Successive Halving
	2. Hyperband
	3. BOHB

5. **Other Search Algorthms**
	1. Simulated Annealing
	2. Population Based Optimization

6. **Gentetic Algorithms**
	1. CMA-ES	

7. **Python tools**
	1. Scikit-learn
	2. Scikit-optimize
	3. Hyperopt
	4. Optuna
	5. Keras Tuner
	6. SMAC
	7. Others

## Links

- [Online Course](https://www.trainindata.com/p/hyperparameter-optimization-for-machine-learning)
