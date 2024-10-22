import os
from pathlib import Path

PROJECT_PATH = str(Path(__file__).parent.parent.parent.parent)

PATH_TO_DATA = str(Path(Path(__file__).parent.parent.parent.parent, "datasets"))

DEFAULT_PATH_RESULTS = os.path.join(PROJECT_PATH, "results_of_experiments")


def data_path(dataset_name: str, log: bool = False):
    if log:
        print("Data Path: ", Path(PATH_TO_DATA, dataset_name))
    return Path(PATH_TO_DATA, dataset_name)


YOLO_DATA_URL = "https://ultralytics.com/assets/"


def yolo_data_path(dataset_name: str):
    data_name = f"{dataset_name}.zip"
    return Path(PATH_TO_DATA, dataset_name, data_name)


YOLO_YAML_URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/19a2d0a4b09f9509629caf89ca6fb8118dc9ba5d/ultralytics/cfg/datasets/"


def yolo_yaml_path(dataset_name: str):
    yaml_name = f"{dataset_name}.yaml"
    return Path(PATH_TO_DATA, dataset_name, yaml_name)
