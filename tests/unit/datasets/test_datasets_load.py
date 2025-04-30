import os
import pytest

from fedcore.architecture.dataset.datasets_from_source import (
    AbstractDataset, 
    ObjectDetectionDataset, 
    TimeSeriesDataset
    )
from fedcore.architecture.utils.paths import PATH_TO_DATA


def image_dataset():
    image_clf_source = os.path.join(PATH_TO_DATA, 'image_segmentation', "train")
    return AbstractDataset(data_source=image_clf_source)


def od_dataset():
    image_od_source = os.path.join(PATH_TO_DATA, 'object_detection', 'chips.yaml')
    image_od_annotation = os.path.join(PATH_TO_DATA, 'object_detection', "labels")
    return ObjectDetectionDataset(data_source=image_od_source, annotation_source=image_od_annotation)


def ts_dataset():
    ts_source = os.path.join(PATH_TO_DATA, 'time_series_forecasting', 'multi_dim', 'debet_forecasting',
                             'train.csv')
    return TimeSeriesDataset(data_source=ts_source)

@pytest.mark.skip(reason="datasets aren't in their places yet")
@pytest.mark.parametrize('dataset', (ts_dataset, image_dataset, od_dataset))
def test_load_dataset(dataset):
    
    dataset = dataset()
    image, target = next(iter(dataset))
    assert image is not None
    assert target is not None
