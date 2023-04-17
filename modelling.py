import logging
import time
from typing import List, Dict
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold, ParameterGrid
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import plotly.graph_objs as go
import plotly.io as pio
import plotly.subplots as sp

def train_and_evaluate_multiple_models(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, models: list[str], metrics: list[str] = ["accuracy"], average: str = "weighted") -> dict[str, dict[str, float]]:
    """
    Train and evaluate multiple classifiers on the given data.

    Parameters
    ----------
    X_train: pd.DataFrame
        The features for the training data.

    y_train: pd.Series
        The target variable for the training data.

    X_test: pd.DataFrame
        The features for the testing data.

    y_test: pd.Series
        The target variable for the testing data.

    models: list[str]
        A list of classifiers to use, can include 'XGBoost', 'Random Forest', 'Extra Trees', 'Adaboost'.
        
    metrics: list[str], optional, default: ["accuracy"]
        The evaluation metrics to use, can include 'accuracy', 'precision', 'f1', and 'recall'.

    average: str, optional, default: "weighted"
        The averaging method for multiclass targets, one of [None, 'micro', 'macro', 'weighted'].
    
    Returns
    -------
    results: dict[str, dict[str, float]]
        A dictionary with the model names as keys, then a nested dictionary with metrics as key and the value of the metric as the value.
    """
    model_dict = {
        "XGBoost": XGBClassifier,
        "Random Forest": RandomForestClassifier,
        "Extra Trees": ExtraTreesClassifier,
        "Adaboost": AdaBoostClassifier
    }
    
    metric_dict = {
        "accuracy": accuracy_score,
        "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average=average),
        "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average=average),
        "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average=average)
    }
    
    num_classes = len(y_train.unique())

    results = {}
    for model in models:
        if model not in model_dict:
            raise ValueError(f"Invalid model name: {model}. Must be one of: {list(model_dict.keys())}")

        if model == "XGBoost":
            classifier = model_dict[model](objective='multi:softprob', num_class=num_classes)
        else:
            classifier = model_dict[model]()
        
        logging.info(f"Training {model}...")
        start_time = time.time()
        classifier.fit(X_train, y_train)
        train_time = time.time() - start_time

        logging.info(f"Predicting with {model}...")
        start_time = time.time()
        y_pred = classifier.predict(X_test)
        predict_time = time.time() - start_time
        
        model_results = {
            "train_time": train_time,
            "predict_time": predict_time
        }
        for metric in metrics:
            if metric not in metric_dict:
                raise ValueError(f"Invalid metric name: {metric}. Must be one of: {list(metric_dict.keys())}")
            model_results[metric] = metric_dict[metric](y_test, y_pred)
        
        results[model] = model_results
        logging.info(f"{model} evaluation complete.\n")

    return results

def model_performance_report(model, y_test: pd.Series, y_pred: pd.Series) -> None:
    """
    Visualize model performance using a confusion matrix and metrics.

    Parameters
    ----------
    model: estimator instance
        The classifier model instance.
        
    y_test: pd.Series
        The true target variable for the testing data.

    y_pred: pd.Series
        The predicted target variable for the testing data.
    """
    # Create confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    trace1 = go.Heatmap(z=conf_matrix,
                        x=["0 (pred)", "1 (pred)", "2 (pred)"],
                        y=["0 (true)", "1 (true)", "2 (true)"],
                        xgap=2, ygap=2,
                        colorscale='Viridis', showscale=False)

    # Calculate metrics
    Accuracy = accuracy_score(y_test, y_pred)
    Precision = precision_score(y_test, y_pred, average='weighted')
    Recall = recall_score(y_test, y_pred, average='weighted')
    F1_score = f1_score(y_test, y_pred, average='weighted')

    show_metrics = pd.DataFrame(data=[[Accuracy, Precision, Recall, F1_score]])
    show_metrics = show_metrics.T

    colors = ['gold', 'lightgreen', 'lightcoral', 'lightskyblue']
    trace2 = go.Bar(x=show_metrics[0].values,
                    y=['Accuracy', 'Precision', 'Recall', 'F1_score'],
                    text=np.round_(show_metrics[0].values, 4),
                    textposition='auto',
                    orientation='h', opacity=0.8,
                    marker=dict(
                        color=colors,
                        line=dict(color='#000000', width=1.5)))

    # Create subplots
    fig = sp.make_subplots(rows=2, cols=1, print_grid=False,
                           subplot_titles=('Confusion Matrix',
                                           'Metrics',
                                           ))

    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 2, 1)

    fig.update_layout(showlegend=False,
                      title='<b>Model performance report</b><br>' + str(model),
                      autosize=True, height=800, width=800,
                      plot_bgcolor='rgba(240,240,240, 0.95)',
                      paper_bgcolor='rgba(240,240,240, 0.95)',
                      )
    fig.layout.titlefont.size = 14

    pio.show(fig, filename='model-performance')

def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models: list[str] = ["XGBoost", "Random Forest", "Extra Trees", "Adaboost"],
    scoring: str = "accuracy",
    cv_type: str = "KFold",
    n_splits: int = 5,
    tuning_type: str = "GridSearchCV",
    xgb_params: dict = None,
    rf_params: dict = None,
    et_params: dict = None,
    ab_params: dict = None,
) -> dict[str, dict[str, float]]:
    """
    Automatically perform hyperparameter tuning for the selected models.

    Parameters
    ----------
    X_train: pd.DataFrame
        The features for the training data.

    y_train: pd.Series
        The target variable for the training data.

    models: list[str], optional, default: ["XGBoost", "Random Forest", "Extra Trees", "Adaboost"]
        A list of classifiers to use, can include 'XGBoost', 'Random Forest', 'Extra Trees', 'Adaboost'.

    scoring: str, optional, default: "accuracy"
        The scoring metric to be used for hyperparameter tuning.

    cv_type: str, optional, default: "KFold"
        The type of cross-validation to use, one of ["KFold", "StratifiedKFold"].

    tuning_type: str, optional, default: "GridSearchCV"
        The type of hyperparameter tuning to use, one of ["GridSearchCV", "RandomizedSearchCV"].

    xgb_params: dict, optional
        A dictionary of hyperparameters for the XGBoost model.

    rf_params: dict, optional
        A dictionary of hyperparameters for the Random Forest model.

    et_params: dict, optional
        A dictionary of hyperparameters for the Extra Trees model.

    ab_params: dict, optional
        A dictionary of hyperparameters for the Adaboost model.

    Returns
    -------
    results: dict[str, dict[str, float]]
        A dictionary with the model names as keys, then a nested dictionary with the keys 'best_params_' and 'best_score_' and their corresponding values as the values.
    """
    model_dict = {
        "XGBoost": XGBClassifier,
        "Random Forest": RandomForestClassifier,
        "Extra Trees": ExtraTreesClassifier,
        "Adaboost": AdaBoostClassifier,
    }

    param_dict = {
        "XGBoost": xgb_params,
        "Random Forest": rf_params,
        "Extra Trees": et_params,
        "Adaboost": ab_params,
    }

    if cv_type == "KFold":
        cv = KFold(n_splits=n_splits)
    elif cv_type == "StratifiedKFold":
        cv = StratifiedKFold(n_splits=n_splits)
    else:
        raise ValueError(f"Invalid cv_type: {cv_type}. Must be one of: ['KFold', 'StratifiedKFold']")

    if tuning_type == "GridSearchCV":
        tuning_function = GridSearchCV
    elif tuning_type == "RandomizedSearchCV":
        tuning_function = RandomizedSearchCV
    else:
        raise ValueError(f"Invalid tuning_type: {tuning_type}. Must be one of: ['GridSearchCV', 'RandomizedSearchCV']")

    results = {}
    for model in models:
        if model not in model_dict:
            raise ValueError(f"Invalid model name: {model}. Must be one of: {list(model_dict.keys())}")

        classifier = model_dict[model]()
        params = param_dict[model]

        if params is None:
            raise ValueError(f"Hyperparameters for {model} are not provided. Please provide a dictionary of hyperparameters for the model.")

        tuner = tuning_function(classifier, params, scoring=scoring, cv=cv)

        total_candidates = len(list(ParameterGrid(params)))
        total_fits = n_splits * total_candidates
        logging.info(f"Fitting {n_splits} folds for each of {total_candidates} candidates, totalling {total_fits} fits")

        logging.info(f"Tuning hyperparameters for {model}...")
        start_time = time.time()
        tuner.fit(X_train, y_train)
        tuning_time = time.time() - start_time

        model_results = {
            "best_params_": tuner.best_params_,
            "best_score_": tuner.best_score_,
            "tuning_time": tuning_time
        }

        results[model] = model_results
        logging.info(f"Hyperparameter tuning for {model} complete.\n")

    return results