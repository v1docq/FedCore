def evaluate_original_model(fedcore_compressor, input_data):
    original_prediction = fedcore_compressor.predict(input_data, output_mode="default")
    original_output = original_prediction.predict
    original_model = fedcore_compressor.original_model
    original_quality_metrics = fedcore_compressor.evaluate_metric(
        predicton=original_output, target=fedcore_compressor.target
    )
    original_inference_metrics = fedcore_compressor.evaluate_metric(
        predicton=original_output,
        target=fedcore_compressor.target,
        metric_type="original_computational",
    )
    return dict(
        original_model=original_model,
        quality_metrics=original_quality_metrics,
        inference_metrics=original_inference_metrics,
    )


def evaluate_optimised_model(fedcore_compressor, input_data):
    low_rank_prediction = fedcore_compressor.predict(input_data, output_mode='fedcore')
    low_rank_output = low_rank_prediction.predict
    low_rank_model = fedcore_compressor.optimised_model
    low_rank_quality_metrics = fedcore_compressor.evaluate_metric(
        predicton=low_rank_output, target=fedcore_compressor.target
    )
    low_rank_inference_metrics = fedcore_compressor.evaluate_metric(
        predicton=low_rank_output,
        target=fedcore_compressor.target,
        metric_type="optimised_computational",
    )
    return dict(
        optimised_model=low_rank_model,
        quality_metrics=low_rank_quality_metrics,
        inference_metrics=low_rank_inference_metrics,
    )


import torch
from torch import nn
from fedot.core.data.data import OutputData
from fedot.core.repository.dataset_types import DataTypesEnum


def evaluation_loader(data, split='val'):
    source = getattr(data, f'{split}_dataloader', None)
    if source is None and hasattr(data, 'features'):
        source = getattr(data.features, f'{split}_dataloader', None)
    if source is None:
        raise ValueError(f'{split}_dataloader is required for prediction')
    return source


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_to_device(v, device) for v in value)
    return value


def predict_modules(models, data, split='val'):
    """Evaluate all models on each batch, with one aligned target/identity stream.

    Single-pass and shuffled sources are supported without re-reading them.
    Each identity is the object's occurrence offset within this evaluation.
    """
    if not all(isinstance(model, nn.Module) for model in models.values()):
        raise TypeError('Paired evaluation requires torch.nn.Module models')
    modes = {name: [(child, child.training) for child in model.modules()]
             for name, model in models.items()}
    predictions, targets, identifiers = {name: [] for name in models}, [], []
    offset = 0
    try:
        for model in models.values():
            model.eval()
        with torch.inference_mode():
            for batch in evaluation_loader(data, split):
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    raise TypeError('Expected labeled batches (inputs, target[, object_ids])')
                inputs, target = batch[:2]
                target = torch.as_tensor(target).detach().cpu()
                if target.ndim == 0:
                    raise ValueError('Targets must retain the batch dimension')
                ids = (torch.as_tensor(batch[2]).detach().cpu() if len(batch) > 2
                       else torch.arange(offset, offset + len(target)))
                if len(ids) != len(target):
                    raise ValueError('Object identifiers and targets have different lengths')
                targets.append(target)
                identifiers.append(ids)
                offset += len(target)
                for name, model in models.items():
                    device = next(model.parameters(), torch.empty(0)).device
                    values = _to_device(inputs, device)
                    result = model(**values) if isinstance(values, dict) else (
                        model(*values) if isinstance(values, (list, tuple)) else model(values))
                    result = getattr(result, 'logits', result)
                    if not isinstance(result, torch.Tensor) or result.shape[0] != len(target):
                        raise ValueError('Model output must be a tensor aligned with the batch')
                    predictions[name].append(result.detach().cpu())
    finally:
        for states in modes.values():
            for child, training in states:
                child.training = training
    if not targets:
        raise ValueError('Evaluation source is empty')
    return (torch.cat(identifiers), torch.cat(targets),
            {name: torch.cat(parts) for name, parts in predictions.items()})


def predict_module(model, data, split='val'):
    ids, targets, predictions = predict_modules({'model': model}, data, split)
    return OutputData(idx=ids.numpy(), task=data.task, predict=predictions['model'],
                      target=targets, data_type=DataTypesEnum.table)
