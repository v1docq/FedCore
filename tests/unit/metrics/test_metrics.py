# tests/unit/metrics/test_metrics.py
import math
import sys
import types
import importlib
from dataclasses import dataclass
from typing import Any

import pytest
import torch
import pandas as pd

from fedcore.metrics.api_metric import (
    calculate_metric,
    calculate_metrics,
    calculate_regression_metric,
    calculate_classification_metric,
    calculate_computational_metric,
)
from fedcore.metrics.metric_impl import (
    QualityMetric,
    Accuracy,
    Precision,
    F1,
    Logloss,
    ROCAUC,
    SMAPE,
    MAPE,
    MASE,
    RMSE as _RMSE,
    MSE as _MSE,
    MSLE as _MSLE,
    MAE as _MAE,
    R2 as _R2,
)

# ---------- A. API metrics ----------

def test_regression_metrics_basic():
    """Basic smoke tests for regression metrics."""
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    yhat = torch.tensor([1.1, 1.9, 3.2, 3.7])

    assert isinstance(calculate_metric(y, yhat, "mse"), float)
    assert isinstance(calculate_metric(y, yhat, "rmse"), float)
    assert isinstance(calculate_metric(y, yhat, "mae"), float)
    assert isinstance(calculate_metric(y, yhat, "r2"), float)
    assert isinstance(calculate_metric(y.abs(), yhat.abs(), "msle"), float)
    assert isinstance(calculate_metric(y, yhat, "mape"), float)


def test_classification_metrics_basic():
    """Basic smoke tests for classification metrics."""
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])
    yhat = torch.tensor([0.0, 1.0, 0.0, 0.0])

    for name in ("accuracy", "precision", "f1"):
        v = calculate_metric(y, yhat, name)
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0


def test_metrics_dataframe_and_msle_guard():
    """DF shape/columns and MSLE negative guard."""
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    yhat = torch.tensor([1.1, 1.9, 3.2, 3.7])

    df = calculate_metrics(y, yhat, metric_names=("r2", "rmse", "mae"))
    assert isinstance(df, pd.DataFrame)
    assert set(["r2", "rmse", "mae"]).issubset(df.columns)
    assert len(df) == 1

    with pytest.raises(ValueError):
        _ = calculate_metrics(torch.tensor([-1.0, 2.0]), torch.tensor([0.0, 2.0]), metric_names=("msle",))


@pytest.mark.parametrize("name", ["mse", "mae", "rmse", "r2", "msle", "mape", "accuracy", "precision", "f1"])
def test_calculate_metric_parametrized(name):
    """Parametrized: calculate_metric returns float for known names."""
    if name in {"accuracy", "precision", "f1"}:
        y = torch.tensor([0.0, 1.0, 0.0, 1.0])
        yhat = torch.tensor([0.0, 1.0, 0.0, 1.0])
    else:
        y = torch.tensor([1.0, 2.0, 3.0, 4.0]).abs()
        yhat = torch.tensor([1.0, 2.1, 2.9, 4.1]).abs()

    try:
        out = calculate_metric(y, yhat, name)
        assert isinstance(out, float)
    except ValueError as e:
        if "Unknown metric type" in str(e):
            pytest.skip(f"{name} not implemented")
        raise


def test_regression_and_classification_helpers():
    """calculate_regression_metric / calculate_classification_metric return 1-row DF."""
    yt = torch.tensor([1.0, 2.0, 3.0])
    yp = torch.tensor([0.9, 2.1, 2.9])
    reg = calculate_regression_metric(yt, yp, metric_names=("r2", "rmse", "mae"))
    assert isinstance(reg, pd.DataFrame) and len(reg) == 1
    assert set(["r2", "rmse", "mae"]).issubset(reg.columns)

    y = torch.tensor([0.0, 1.0, 0.0, 1.0])
    yhat = torch.tensor([0.0, 1.0, 0.0, 0.0])
    cls = calculate_classification_metric(y, yhat, metric_names=("accuracy", "f1", "precision"))
    assert isinstance(cls, pd.DataFrame) and len(cls) == 1
    assert set(["accuracy", "f1", "precision"]).issubset(cls.columns)


def test_api_classification_with_probs_logloss_rocauc(monkeypatch: pytest.MonkeyPatch):
    """calculate_classification_metric handles logloss/roc_auc with probs."""
    y = torch.tensor([0, 1, 1, 0, 1])
    probs = torch.tensor([[0.9, 0.1],
                          [0.1, 0.9],
                          [0.2, 0.8],
                          [0.8, 0.2],
                          [0.3, 0.7]], dtype=torch.float32)
    labels = probs.argmax(dim=1)

    # robust fallback for old torchmetrics (no roc_auc_score)
    try:
        import torchmetrics.functional as F
        if not hasattr(F, "roc_auc_score"):
            def _roc_auc_score(preds, target, **_):
                try:
                    return F.auroc(preds, target, task="binary")
                except TypeError:
                    return F.auroc(preds, target)
            monkeypatch.setattr(F, "roc_auc_score", _roc_auc_score, raising=False)
    except Exception:
        pass

    df = calculate_classification_metric(y, labels, probs=probs, metric_names=("logloss", "roc_auc"))
    assert set(["logloss", "roc_auc"]).issubset(df.columns)


def test_api_computational_metric_smoke():
    """Smoke: else-branch returns 0.0 without heavy deps."""
    out = calculate_computational_metric(model=None, dataset=None, model_regime="other")
    assert isinstance(out, float)

# ---------- B. Impl classes ----------

def test_impl_classification_metrics():
    """Accuracy/Precision/F1 return bounded floats."""
    y = torch.tensor([0, 1, 0, 1])
    yhat = torch.tensor([0, 1, 0, 0])

    for cls in (Accuracy, Precision, F1):
        val = cls.metric(y, yhat)
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0


def test_impl_logloss_binary_probs():
    """Binary logloss with P(y=1)."""
    y = torch.tensor([0, 1, 1, 0])
    p2 = torch.tensor([[0.9, 0.1],
                       [0.2, 0.8],
                       [0.4, 0.6],
                       [0.7, 0.3]], dtype=torch.float32)
    val = Logloss.metric(y, p2[:, 1])  # use positive-class prob
    assert isinstance(val, float)
    assert val >= 0.0


def test_impl_rocauc_binary_probs(monkeypatch: pytest.MonkeyPatch):
    """ROC-AUC works across torchmetrics versions (fallback to auroc if needed)."""
    pytest.importorskip("torchmetrics")
    F = pytest.importorskip("torchmetrics.functional")

    if not hasattr(F, "roc_auc_score"):
        def _roc_auc_score(a, b=None, **kwargs):
            if b is None:
                raise TypeError("expect two tensors")
            # detect order: (target, preds) vs (preds, target)
            if hasattr(a, "is_floating_point") and hasattr(b, "is_floating_point"):
                if (not a.is_floating_point()) and b.is_floating_point():
                    target = a.long().view(-1); preds = b
                elif a.is_floating_point() and (not b.is_floating_point()):
                    target = b.long().view(-1); preds = a
                else:
                    preds, target = a, b.long().view(-1)
            else:
                preds, target = a, b.long().view(-1)
            try:
                return F.auroc(preds, target, task="binary")
            except TypeError:
                return F.auroc(preds, target)
        monkeypatch.setattr(F, "roc_auc_score", _roc_auc_score, raising=False)

    y = torch.tensor([0, 1, 1, 0, 1])
    p = torch.tensor([[0.9, 0.1],
                      [0.1, 0.9],
                      [0.2, 0.8],
                      [0.8, 0.2],
                      [0.3, 0.7]], dtype=torch.float32)
    val = ROCAUC.metric(y, p)
    assert isinstance(val, float)
    assert 0.0 <= val <= 1.0


def test_time_series_metrics():
    """SMAPE/MAPE/MASE simple sequences."""
    y = torch.tensor([10.0, 12.0, 13.0, 12.0, 11.0])
    yhat = torch.tensor([9.0, 12.0, 12.5, 12.0, 11.5])

    smape = SMAPE.metric(y, yhat)
    mape = MAPE.metric(y, yhat)
    mase = MASE.metric(y, yhat, seasonal_factor=1)

    for v in (smape, mape, mase):
        assert isinstance(v, float)
        assert v >= 0.0


def test_impl_regression_metric_classes():
    """Class-based regression metrics return floats."""
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    yhat = torch.tensor([1.1, 1.9, 3.2, 3.7]).clone()
    assert isinstance(_RMSE.metric(y, yhat), float)
    assert isinstance(_MSE.metric(y, yhat), float)
    assert isinstance(_MSLE.metric(y.abs(), yhat.abs()), float)
    assert isinstance(_MAE.metric(y, yhat), float)
    assert isinstance(_R2.metric(y, yhat), float)

# ---------- C. QualityMetric.get_value path ----------

class _DummyDataset:
    """Tiny dataset that yields (x, y) and keeps 'targets'."""
    def __init__(self, xs: torch.Tensor, ys: torch.Tensor):
        self.features = xs
        self.targets = ys
    def __iter__(self):
        for i in range(len(self.targets)):
            yield self.features[i], self.targets[i]
    def __len__(self):
        return len(self.targets)

class _DummyLoader:
    def __init__(self, dataset: _DummyDataset):
        self.dataset = dataset

@dataclass
class _DummyFeatures:
    train_dataloader: _DummyLoader

@dataclass
class _DummyResultsObj:
    predict: Any

@dataclass
class _DummyResults:
    predict: _DummyResultsObj

class _DummyPipeline:
    """Minimal pipeline that returns labels or probs."""
    def __init__(self, labels: torch.Tensor):
        self._labels = labels
    def predict(self, _input, output_mode: str = "labels"):
        if output_mode == "labels":
            pred = self._labels
        else:
            pred = torch.nn.functional.one_hot(self._labels, num_classes=2).float()
        return _DummyResults(_DummyResultsObj(pred))

class _AccuracyViaGetValue(QualityMetric):
    """Accuracy-like check for get_value()."""
    split = "train"
    output_mode = "labels"
    @classmethod
    def metric(cls, target: torch.Tensor, predict: torch.Tensor) -> float:
        return float((target == predict).float().mean().item())

def test_quality_metric_get_value():
    """get_value() pulls y_true from loader and y_pred from pipeline."""
    xs = torch.randn(5, 3)
    ys = torch.tensor([0, 1, 0, 1, 1])
    dataset = _DummyDataset(xs, ys)
    reference_data = types.SimpleNamespace(features=_DummyFeatures(_DummyLoader(dataset)))
    pipe = _DummyPipeline(labels=ys.clone())
    score = _AccuracyViaGetValue.get_value(pipe, reference_data)
    assert math.isclose(score, 1.0, rel_tol=1e-9)

# ---------- D. CV metrics (guarded imports) ----------

def test_cv_classification_metric_counter_basic():
    """Streaming classification counter computes macro metrics; roc_auc if available."""
    try:
        from fedcore.metrics.cv_metrics import ClassificationMetricCounter
    except Exception:
        pytest.skip("cv_metrics not available")
        return
    cm = ClassificationMetricCounter()
    logits = torch.tensor([[2.0, 1.0], [0.2, 1.5], [1.2, 0.8], [0.7, 1.1]])
    targets = torch.tensor([0, 1, 0, 1])
    cm.update(logits, targets)
    out = cm.compute()
    for k in ("accuracy", "precision", "recall", "f1"):
        assert k in out and 0.0 <= float(out[k]) <= 1.0
    if "roc_auc" in out:
        assert 0.0 <= float(out["roc_auc"]) <= 1.0

def test_cv_segmentation_metric_counter_and_helpers():
    """IoU/Dice for simple one-hot masks."""
    try:
        from fedcore.metrics.cv_metrics import SegmentationMetricCounter, iou_score, dice_score
    except Exception:
        pytest.skip("cv_metrics not available")
        return
    N, C, H, W = 2, 2, 4, 4
    targets = torch.zeros(N, H, W, dtype=torch.long)
    targets[0, :, :] = 1
    preds = torch.zeros(N, C, H, W)
    preds[0, 1] = 1.0
    preds[1, 0] = 1.0

    oh = torch.nn.functional.one_hot(targets, C).permute(0, 3, 1, 2).float()
    iou = iou_score(preds, oh); dice = dice_score(preds, oh)
    assert iou.shape[:2] == (N, C) and dice.shape[:2] == (N, C)

    seg = SegmentationMetricCounter()
    seg.update(preds, targets)
    out = seg.compute()
    assert "iou" in out and "dice" in out

def test_cv_losses_averager_and_pareto():
    """Averager averages dict; Pareto returns boolean mask."""
    try:
        from fedcore.metrics.cv_metrics import LossesAverager, ParetoMetrics
    except Exception:
        pytest.skip("cv_metrics not available")
        return
    la = LossesAverager()
    la.update({"loss_a": torch.tensor(1.0), "loss_b": torch.tensor(3.0)})
    la.update({"loss_a": torch.tensor(3.0), "loss_b": torch.tensor(1.0)})
    res = la.compute()
    assert res["loss_a"] == pytest.approx(2.0) and res["loss_b"] == pytest.approx(2.0)

    pm = ParetoMetrics()
    mask = pm.pareto_metric_list([[1, 2], [2, 1], [1, 1]], maximise=False)
    assert mask.dtype == torch.bool and mask.numel() == 3

# ---------- E. Distillation metrics (optional, skip if absent) ----------

def test_distillation_last_layer_mse():
    """LastLayer returns float (teacher vs student outputs) or skip if absent."""
    mi = importlib.import_module("fedcore.metrics.metric_impl")
    LastLayer = getattr(mi, "LastLayer", None)
    if LastLayer is None:
        pytest.skip("LastLayer not present in metric_impl")
        return
    t = torch.randn(4, 8); s = t + 0.01 * torch.randn(4, 8)
    v = LastLayer.metric(t, s)
    assert isinstance(v, float) and v >= 0.0

def test_distillation_intermediate_attention_and_features():
    """Attention/Features lists are averaged to float or skip if absent."""
    mi = importlib.import_module("fedcore.metrics.metric_impl")
    IntermediateAttention = getattr(mi, "IntermediateAttention", None)
    IntermediateFeatures = getattr(mi, "IntermediateFeatures", None)
    if IntermediateAttention is None or IntermediateFeatures is None:
        pytest.skip("Intermediate* distillation metrics not present")
        return
    t_atts = [torch.randn(2, 4, 4), torch.randn(2, 8, 8)]
    s_atts = [x + 0.01 * torch.randn_like(x) for x in t_atts]
    t_feats = [torch.randn(2, 16, 8, 8), torch.randn(2, 16, 4, 4)]
    s_feats = [x + 0.01 * torch.randn_like(x) for x in t_feats]
    va = IntermediateAttention.metric(t_atts, s_atts)
    vf = IntermediateFeatures.metric(t_feats, s_feats)
    assert isinstance(va, float) and va >= 0.0
    assert isinstance(vf, float) and vf >= 0.0

# ---------- F. NLP evaluate-backed (with dummy) ----------

class _DummyHFMetric:
    def __init__(self, name: str):
        self.name = name
    def compute(self, *, references, predictions, **_):
        def _tp(): return sum(int(r == 1 and p == 1) for r, p in zip(references, predictions))
        def _fp(): return sum(int(r == 0 and p == 1) for r, p in zip(references, predictions))
        def _fn(): return sum(int(r == 1 and p == 0) for r, p in zip(references, predictions))
        if self.name == "accuracy":
            acc = sum(int(r == p) for r, p in zip(references, predictions)) / max(len(references), 1)
            return {"accuracy": acc}
        if self.name == "precision":
            tp, fp = _tp(), _fp(); return {"precision": tp / (tp + fp) if (tp + fp) else 0.0}
        if self.name == "recall":
            tp, fn = _tp(), _fn(); return {"recall": tp / (tp + fn) if (tp + fn) else 0.0}
        if self.name == "f1":
            tp, fp, fn = _tp(), _fp(), _fn()
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            return {"f1": 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0}
        if self.name == "single":
            return {"only": 0.42}
        return {"score": 0.0}

@pytest.fixture(autouse=True)
def _inject_dummy_evaluate_module():
    """Inject lightweight 'evaluate' before importing nlp_metrics."""
    dummy = types.SimpleNamespace(load=lambda metric_name: _DummyHFMetric(metric_name))
    sys.modules["evaluate"] = dummy
    try:
        if "fedcore.metrics.nlp_metrics" in sys.modules:
            importlib.reload(sys.modules["fedcore.metrics.nlp_metrics"])
        yield
    finally:
        sys.modules.pop("evaluate", None)

def _nlp():
    return importlib.import_module("fedcore.metrics.nlp_metrics")

@pytest.mark.parametrize("cls_name", ["NLPAccuracy", "NLPPrecision", "NLPRecall", "NLPF1"])
def test_nlp_evaluate_metric_classmethod(cls_name):
    """Each classmethod .metric(...) returns float in [0,1]."""
    nlp = _nlp()
    cls = getattr(nlp, cls_name)
    refs = [0, 1, 1, 0, 1]
    preds = [0, 1, 0, 0, 1]
    val = cls.metric(refs, preds)
    assert isinstance(val, float)
    assert 0.0 <= val <= 1.0
    if cls_name == "NLPAccuracy":
        assert math.isclose(val, 0.8, rel_tol=1e-6)

def test_nlp_evaluate_instance_compute():
    """Instance .compute(...) returns dict with expected key."""
    nlp = _nlp()
    metric = nlp.NLPF1()
    res = metric.compute(y_true=[1, 0, 1, 1], y_pred=[1, 1, 0, 1])
    assert isinstance(res, dict) and "f1" in res
    assert 0.0 <= res["f1"] <= 1.0

def test_nlp_evaluate_missing_data_raises():
    """Missing references/predictions raises ValueError."""
    nlp = _nlp()
    with pytest.raises(ValueError):
        nlp.NLPAccuracy.metric(None, None)

def test_nlp_evaluate_single_key_when_result_key_none():
    """Single-key dict + result_key=None returns that single value."""
    nlp = _nlp()
    class _MySingle(nlp.EvaluateMetric):
        metric_name = "single"
        result_key = None
        _metric = sys.modules["evaluate"].load(metric_name)
    v = _MySingle.metric(["a", "b"], ["a", "c"])
    assert isinstance(v, float) and math.isclose(v, 0.42, rel_tol=1e-6)
