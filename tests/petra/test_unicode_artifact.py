"""A real executable artifact under the user's Unicode workspace layout."""
import torch
from torch import nn

from fedcore.experiments.measurement import measure_artifact, predict_loaded_artifact


def test_torchscript_export_reload_and_quality_in_unicode_directory(tmp_path):
    torch.manual_seed(17)
    model = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2)).eval()
    x = torch.randn(7, 3)
    record = measure_artifact(model, x[:3], tmp_path / 'данные_эксперимента' / 'модель',
                              repeats=2, warmup=1)
    assert record['status'] == 'succeeded', record
    prediction = predict_loaded_artifact(record['artifact'], x)
    torch.testing.assert_close(prediction, model(x))
    assert len(record['raw_inference_ms']) == 2
