"""
.. module:: modeling

modeling
******

:Description: modeling

    After preprocessing the data into test and train, 
    we apply different modeling techniques and check which give best results.
    This script is given a large hyperparameter search space with various models defined as a
    dictionary ('params'). It will train each of these models and parameters doing grid-search,
    and once the best hyperparameter is found, it returns the test metrics.

    This script does basically the same as what sklearn's "pipeline" does, however i discovered it
    after i did it :')

:Authors:
    benjami parellada

:Version: 

:Date:  
"""

__author__ = 'benjami parellada'

import pickle, sys, os, warnings, itertools
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,cross_val_score

from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

def generate_layers(sizes):
    """
    Function that generates all the combinations of the vector 'sizes' of length > 1
    for the hidden layers of the MLPClassifier
    """
    import itertools
    layers = []
    for L in range(0, len(sizes)+1):
        for subset in itertools.combinations(sizes, L):
            layers.append(subset)
    return layers[len(sizes)+1:] # remove all size 1

seed = 94

# Dictionary of the hyperparameters searched

## Full Grid Search, do not run. takes way too long
params = {
    'DummyClassifier': {'strategy': ['most_frequent']},
    'LogisticRegression': {'solver': ['saga'], 'max_iter': [100000], 'penalty': ['l1', 'l2', 'elasticnet', 'none'], 'C': [0.001, 0.1, 0.5, 1, 2, 10], 'l1_ratio': [0.2, 0.4, 0.5, 0.6, 0.8], 'random_state': [seed]},
    'LinearDiscriminantAnalysis': {},
    'QuadraticDiscriminantAnalysis': {'reg_param': [0.2, 0.3, 0.4, 0.6, 0.7, 0.8]},
    'KNeighborsClassifier': {'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 21, 55, 101], 'weights': ['uniform', 'distance'], 'metric': ['minkowski', 'chebyshev'], 'p': [1, 2, 3, 4, 1.5, 2.5, 3.5]},
    'LinearSVC': {'penalty': ['l1', 'l2'], 'loss': ['hinge', 'squared_hinge'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'max_iter': [1e9], 'random_state': [seed]},
    'SVC': {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'kernel': ['poly', 'rbf', 'sigmoid'], 'gamma': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10], 'degree': [2, 3, 4, 5], 'random_state': [seed]},
    'DecisionTreeClassifier': {'criterion': ['gini', 'entropy'], 'max_features': ['sqrt', 'log2', None], 'max_depth': [3, 5, 7, 9, 15, 21, 35, 50, 100, None], 'min_samples_split': [2, 4, 6, 8, 10], 'min_samples_leaf': [1, 2, 4, 6, 8, 10], 'random_state': [seed]},
    'RandomForestClassifier': {'n_estimators': [50, 100, 200, 250, 500, 750], 'max_features': ['sqrt', 'log2', None], 'bootstrap': [True], 'oob_score': [True], 'criterion': ['gini', 'entropy'], 'max_depth': [3, 5, 7, 9, None], 'min_samples_split': [4, 6, 8, 10], 'min_samples_leaf': [1, 2, 4, 6, 8], 'random_state': [seed]},
    'ExtraTreesClassifier': {'n_estimators': [50, 100, 200, 250, 500, 750], 'max_features': ['sqrt', 'log2', None], 'bootstrap': [True], 'oob_score': [True], 'criterion': ['gini', 'entropy'], 'max_depth': [3, 7, 9, None], 'min_samples_split': [4, 6, 8, 10], 'min_samples_leaf': [1, 2, 4, 6, 8], 'random_state': [seed]},
    'GradientBoostingClassifier': {'loss': ['deviance', 'exponential'], 'max_features': ['sqrt', 'log2', None], 'learning_rate': [0.001, 0.01, 0.1, 0.5, 0.8], 'subsample': [0.3, 0.4, 0.5, 0.6, 0.7], 'n_estimators': [100, 150, 200, 250, 500, 750], 'criterion': ['friedman_mse', 'squared_error'], 'max_depth': [3, 5, 7, 9, None], 'min_samples_split': [2, 4, 6, 8, 10], 'min_samples_leaf': [1, 2, 4, 6, 8, 10], 'random_state': [seed]},
    'MLPClassifier': {'hidden_layer_sizes': generate_layers([2, 5, 10, 20, 50]), 'activation': ['identity', 'logistic', 'tanh', 'relu'], 'alpha': 10.0 ** -np.arange(1, 7), 'learning_rate': ['constant', 'invscaling', 'adaptive'], 'learning_rate_init': [0.0001, 0.001, 0.01, 0.1], 'solver': ['lbfgs'], 'max_iter': [100000], 'random_state': [seed]},
}

# Best hyperparameters found.
# Delete or comment the following dict if you want to run the full grid search.
params = {
    'DummyClassifier': {'strategy': ['most_frequent']},
    'LogisticRegression': {'solver': ['saga'], 'max_iter': [100000], 'penalty': ['elasticnet'], 'C': [1], 'l1_ratio': [0.6], 'random_state': [seed]},
    'LinearDiscriminantAnalysis': {},
    'QuadraticDiscriminantAnalysis': {'reg_param': [0.8]},
    'KNeighborsClassifier': {'n_neighbors': [3], 'weights': ['distance'], 'metric': ['chebyshev']},
    'LinearSVC': {'penalty': ['l2'], 'loss': ['hinge'], 'C': [10], 'max_iter': [1e9], 'random_state': [seed]},
    'DecisionTreeClassifier': {'criterion': ['entropy'], 'max_features': [None], 'max_depth': [7], 'min_samples_split': [6], 'min_samples_leaf': [2], 'random_state': [seed]},
    'RandomForestClassifier': {'n_estimators': [250], 'max_features': [None], 'bootstrap': [True], 'oob_score': [True], 'criterion': ['gini'], 'max_depth': [9], 'min_samples_split': [8], 'min_samples_leaf': [2], 'random_state': [seed]},
    'ExtraTreesClassifier': {'n_estimators': [200], 'max_features': [None], 'bootstrap': [True], 'oob_score': [True], 'criterion': ['gini'], 'max_depth': [None], 'min_samples_split': [4], 'min_samples_leaf': [1], 'random_state': [seed]},
    'GradientBoostingClassifier': {'loss': ['deviance'], 'max_features': [None], 'learning_rate': [0.8], 'subsample': [0.5], 'n_estimators': [200], 'criterion': ['friedman_mse'], 'max_depth': [3], 'min_samples_split': [8], 'min_samples_leaf': [1], 'random_state': [seed]},
    'MLPClassifier': {'hidden_layer_sizes': (5, 20, 50), 'activation': ['tanh'], 'alpha': [1e-06], 'learning_rate': ['constant'], 'learning_rate_init': [0.0001], 'solver': ['lbfgs'], 'max_iter': [100000], 'random_state': [seed]},
}


# For each model, which is the Scaler that should be used.
scalers = {
    'DummyClassifier': 'None',
    'LogisticRegression': 'StandardScaler',
    'LinearDiscriminantAnalysis': 'StandardScaler', # could be 'None' as well
    'QuadraticDiscriminantAnalysis': 'None', # could be 'None' as well
    'KNeighborsClassifier': 'MinMaxScaler',
    'LinearSVC': 'StandardScaler',
    'SVC': 'StandardScaler',
    'DecisionTreeClassifier': 'None',
    'RandomForestClassifier': 'None',
    'ExtraTreesClassifier': 'None',
    'GradientBoostingClassifier': 'None',
    'MLPClassifier': 'StandardScaler'
}

def pickle_results(res):
    """
    Function which serializes the returned value from model_train
    """
    with open("./results/fitted_models.pickle","wb") as f:
        pickle.dump(res, f)

def write_metrics(model, metrics):
    """
    Function that saves the metrics in Latex table format, will create the folder ./results if 
    it previously did not exit
    """
    if not os.path.exists('./results/'):
        os.makedirs('./results/')
    if not os.path.exists('./results/metrics.tex'):
        f = open('./results/metrics.tex', 'w')
        for m in metrics:
            f.write('\\textbf{' + m + '} & ')
        f.write('\n')
        f.close()

    f = open('./results/metrics.tex', 'a')
    f.write('\\textbf{' + model + '} & ')
    for m in metrics.values():
        f.write('{0:4f} & '.format(m))
    f.write('\n')
    f.close()

def compute_metrics(y_test, y_pred):
    """
    Function that computes the various score metrics on the test data - prediction
    """
    matrix = confusion_matrix(y_test, y_pred)
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred, average = 'macro'),
        'AUC': roc_auc_score(y_test, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred),
        'TN': matrix[0][0],
        'FP': matrix[0][1],
        'FN': matrix[1][0],
        'TP': matrix[1][1],
    }

def model_train(model, hyperparameters, x_tr, y_tr, x_te, y_te):
    """
    Function that trains a model:
        - It grid-searchs the hyperparameter space given
        - Once the best parameters are found, it recomputes the CV error
        - Once the CV error is found, and the test error is computed

    :param model: the model we wish to train in string format
    :param hyperparameters: dict of the possible parameters to gridsearch
    :param x_tr: the train dataset, already scaled 
    :param y_tr: the target feature for the train. 
    :param x_tr: the train dataset, already scaled 
    :param y_tr: the target feature for the train.
    :return: The fitted regressor, the test-train metrics, the predictions for the test, the best parameters found
    """
    print(f"Fitting {model.__name__}")
    regressor = model()
    best_params_ = dict()
    if hyperparameters: # if the model has hyperparameters
        # Model Regressor parameter search using CV
        grid = GridSearchCV(regressor, hyperparameters, cv=5, n_jobs=-1, scoring = 'recall', verbose = 10)
        params = grid.fit(x_tr, y_tr)
        regressor = model(**params.best_params_).fit(x_tr, y_tr) # refit with the best model
        best_params_ = params.best_params_ # save best params for output
    else:
        regressor.fit(x_tr, y_tr)
    acc_cv = cross_val_score(regressor, x_tr, y_tr, cv=5, scoring = 'accuracy') # accurate estimate of train accuracy
    rec_cv = cross_val_score(regressor, x_tr, y_tr, cv=5, scoring = 'recall') # accurate estimate of train accuracy

    # Evaluation of Model on test data
    y_pr = regressor.predict(x_te) # predict test values
    metrics = compute_metrics(y_te, y_pr) 
    
    metrics['Accuracy Train'] = acc_cv.mean()
    metrics['Recall Train'] = rec_cv.mean()
    return regressor, metrics, y_pr, best_params_

seed = 1984

def read_data():
    """
    Function that reads the serialized data from the exploration
    """
    with open('data/preprocess.pickle', 'rb') as f:
        x_train = pickle.load(f)
        y_train = pickle.load(f)
        x_test = pickle.load(f)
        y_test = pickle.load(f)
    return x_train, y_train, x_test, y_test

def main():
    """
    Main function of the script, given the list of hyperparameters ('params'), it will iterate over it
    and grid-searches each of the presented models with the defined hyperparameters. 
    Once the best parameters have been found, it evaluates it on the train wit CV and then on the test 
    for accurate metrics.
    It stores the results in a dictionary (the fitted regressor, the metrics, the predictions and the best parameters found)
    """

    # there are some hyperparameters that do not work together, these throw a warning. We supress it.
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

    x_train, y_train, x_test, y_test = read_data()
    results = dict()
    for model in params:
        reg, met, y_pr, best = model_train(globals()[model], params[model], x_train[scalers[model]], y_train, x_test[scalers[model]], y_test)
        results[model] = {'regressor': reg, 'metrics': met, 'predictions': y_pr, 'best_params': best}
        write_metrics(model, met)
        pickle_results(results)

if __name__ == '__main__':
    main()
