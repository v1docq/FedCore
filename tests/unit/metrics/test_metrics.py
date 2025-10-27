import pytest 
from fedcore.metrics.quality import MetricFactory, LOADED_METRICS
import torch


label_prob_name_suffix = 'MulticlassPrecision__2'
def test_loading():
    mtr = MetricFactory.get_metric(label_prob_name_suffix)
    assert label_prob_name_suffix in LOADED_METRICS, "For each combination metric - type the corresponding class should be created"
    assert MetricFactory.get_metric(label_prob_name_suffix) is mtr, "In case the previously called metric is used, the existing class should be summoned"


def test_metrics_factory_classification():
    target = torch.Tensor([1, 1,]).long()
    preds = torch.Tensor([[0, 1], [0.2, 0]])
    mtr = MetricFactory.get_metric(label_prob_name_suffix)
    assert mtr.metric(target, preds) is not None, 'wrong `.metric` logic'
    assert mtr.metric(target, torch.argmax(preds, -1)) is not None, 'wrong `.metric` logic'


prob_name_suffix = 'MulticlassAUROC__2'
def test_probs():
    target = torch.Tensor([1, 1,]).long()
    preds = torch.Tensor([[0, 1], [0.2, 0]])
    mtr = MetricFactory.get_metric(prob_name_suffix)
    assert mtr.metric(target, preds) is not None, 'wrong `.metric` logic. the type-based decorator didn\'t cope with the input'


prob_name_suffix = 'MeanSquaredError'
def test_probs():
    target = torch.Tensor([1, 1.])
    preds = torch.Tensor([[0, 1], [0.2, 0]])
    mtr = MetricFactory.get_metric(prob_name_suffix)
    assert mtr.metric(target, preds) is not None, 'wrong `.metric` logic. the type-based decorator didn\'t cope with the input'

