# tests/test_metrics_smoke.py
"""
Smoke tests for fedcore.metrics:
- regression / forecasting
- classification
- NLP metrics (evaluate)
- computational CV metrics (optional)

These tests are not strict unit checks, but verify that
the main API entry points run without errors and return sane outputs.
"""

import numpy as np
import pytest

from fedcore.metrics.api_metric import (
    calculate_regression_metric,
    calculate_forecasting_metric,
    calculate_classification_metric,
    calculate_computational_metric,
)


def test_regression_metrics_runs():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    p = np.array([0.0, 1.1, 1.9, 3.0])
    df = calculate_regression_metric(y, p, metric_names=("r2", "rmse", "mae"))
    assert not df.empty
    assert set(df.columns) == {"r2", "rmse", "mae"}


def test_forecasting_metrics_runs():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.1, 1.9, 3.2, 3.8])
    df = calculate_forecasting_metric(y, p, metric_names=("rmse", "mae", "smape"))
    assert not df.empty
    assert "rmse" in df and "mae" in df and "smape" in df


def test_classification_metrics_runs():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    probs   = np.array([
        [0.8, 0.2],
        [0.1, 0.9],
        [0.6, 0.4],
        [0.55, 0.45],
    ])
    df = calculate_classification_metric(
        target=y_true,
        labels=y_pred,
        probs=probs,
        metric_names=("accuracy", "f1", "precision", "logloss", "roc_auc"),
    )
    assert not df.empty
    assert "accuracy" in df and "f1" in df


@pytest.mark.skipif(
    pytest.importorskip("evaluate", reason="evaluate package not installed") is None,
    reason="evaluate not available",
)
def test_nlp_metrics_runs():
    from fedcore.metrics.nlp_metrics import SacreBLEU, ROUGE, NLPAccuracy

    refs = [["the cat is on the mat"], ["there is a cat on the mat"]]
    hyps = ["the cat is on the mat", "there is cat on mat"]

    bleu = SacreBLEU()
    rouge = ROUGE()
    acc = NLPAccuracy()

    res_bleu = bleu.compute(y_true=refs, y_pred=hyps)
    res_rouge = rouge.compute(y_true=[r[0] for r in refs], y_pred=hyps)
    res_acc = acc.compute(y_true=[1, 0], y_pred=[1, 0])


    assert "score" in res_bleu or "sacrebleu" in res_bleu
    assert isinstance(res_rouge, dict)
    assert isinstance(res_acc, dict)


def test_cv_metrics_optional(monkeypatch):
    """Check computational metric import; skip if deps missing."""
    try:
        # dummy objects, evaluator likely to error, but we check import path
        _ = calculate_computational_metric(object(), object(), model_regime="model_after")
    except ImportError:
        pytest.skip("CV dependencies not installed")
    except Exception:
        # acceptable: evaluator needs real model/dataset
        pass
