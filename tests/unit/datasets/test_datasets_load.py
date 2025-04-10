import unittest
import os

from fedcore.architecture.dataset.datasets_from_source import AbstractDataset, ObjectDetectionDataset, TimeSeriesDataset
from fedcore.architecture.utils.paths import PROJECT_PATH, PATH_TO_DATA


class MyTestCase(unittest.TestCase):
    def __init__(self):
        self.test_scenario = 'time_series'

    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_image_dataset(self):
        image_clf_source = os.path.join(PATH_TO_DATA, 'image_segmentation', "train")
        torch_dataset = AbstractDataset(data_source=image_clf_source)
        return torch_dataset

    def test_od_dataset(self):
        image_od_source = os.path.join(PATH_TO_DATA, 'object_detection', 'chips.yaml')
        image_od_annotation = os.path.join(PATH_TO_DATA, 'object_detection', "labels")
        torch_dataset = ObjectDetectionDataset(data_source=image_od_source,
                                               annotation_source=image_od_annotation)
        return torch_dataset

    def test_ts_dataset(self):
        ts_source = os.path.join(PATH_TO_DATA, 'time_series_regression', 'multi_dim', 'AppliancesEnergy',
                                 'AppliancesEnergy_TRAIN.ts')
        torch_dataset = TimeSeriesDataset(data_source=ts_source)
        return torch_dataset

    def run_test(self):
        scenario_dict = {'time_series': self.test_ts_dataset,
                         'image_segmentation': self.test_image_dataset,
                         'od_yolo_dataset': self.test_od_dataset}
        torch_dataset = scenario_dict[self.test_scenario]()
        image, target = next(iter(torch_dataset))


if __name__ == '__main__':
    unittest.main()
