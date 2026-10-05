"""Offline API illustration; the five labels are a fixture, not an NLP benchmark."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from fedcore.metrics.quality import MetricFactory
from fedcore.experiments.scenarios import causal_token_nll


def main():
    truth = torch.tensor([1, 0, 1, 1, 0])
    predictions = torch.tensor([1, 0, 0, 1, 0])
    for name in ("BinaryAccuracy", "BinaryF1Score"):
        metric = MetricFactory.get_metric(name)
        print(name, float(metric.metric(truth, predictions)))
    logits = torch.zeros(1, 3, 258)
    labels = torch.tensor([[1, 34, 67]])
    print("Uniform causal UTF-8 byte fixture:", causal_token_nll(logits, labels))


if __name__ == "__main__":
    main()
