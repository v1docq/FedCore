import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)
from sklearn.metrics import (
    d2_absolute_error_score,
    explained_variance_score,
    max_error,
    median_absolute_error,
)

from fedcore.metrics.cv_metrics import CV_quality_metric
from fedcore.metrics.metric_impl import (
    Accuracy,
    F1,
    Precision,
    Logloss,
    smape,
    mase,
    mape,
)


def calculate_regression_metric(
    target, labels, rounding_order=3, metric_names=("r2", "rmse", "mae"), **kwargs
):
    target = target.astype(float)

    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    metric_dict = {
        "r2": r2_score,
        "mse": mean_squared_error,
        "rmse": rmse,
        "mae": mean_absolute_error,
        "msle": mean_squared_log_error,
        "mape": mean_absolute_percentage_error,
        "median_absolute_error": median_absolute_error,
        "explained_variance_score": explained_variance_score,
        "max_error": max_error,
        "d2_absolute_error_score": d2_absolute_error_score,
    }

    df = pd.DataFrame(
        {
            name: func(target, labels)
            for name, func in metric_dict.items()
            if name in metric_names
        },
        index=[0],
    )
    return df.round(rounding_order)


def calculate_forecasting_metric(
    target, labels, rounding_order=3, metric_names=("smape", "rmse", "mape"), **kwargs
):
    target = target.astype(float)

    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    metric_dict = {
        "rmse": rmse,
        "mae": mean_absolute_error,
        "median_absolute_error": median_absolute_error,
        "smape": smape,
        "mase": mase,
        "mape": mape,
    }

    df = pd.DataFrame(
        {
            name: func(target, labels)
            for name, func in metric_dict.items()
            if name in metric_names
        },
        index=[0],
    )
    return df.round(rounding_order)


def calculate_classification_metric(
    target, labels, probs, rounding_order=3, metric_names=("f1" "accuracy")
):
    metric_dict = {
        "accuracy": Accuracy,
        "f1": F1,
        "precision": Precision,
        "logloss": Logloss,
    }

    df = pd.DataFrame(
        {
            name: func(target, labels, probs).metric()
            for name, func in metric_dict.items()
            if name in metric_names
        },
        index=[0],
    )
    return df.round(rounding_order)


def calculate_computational_metric(
    target, labels, probs, rounding_order=3, metric_names=("f1" "accuracy")
):
    metric_dict = CV_quality_metric()

    df = pd.DataFrame(
        {
            name: func(target, labels, probs).metric()
            for name, func in metric_dict.items()
            if name in metric_names
        },
        index=[0],
    )
    return df.round(rounding_order)
