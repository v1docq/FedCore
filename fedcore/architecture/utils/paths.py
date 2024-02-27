from pathlib import Path

PROJECT_PATH = str(Path(__file__).parent.parent.parent.parent)

PATH_TO_DATA = str(Path(Path(__file__).parent.parent.parent.parent, 'datasets'))


def data_path(dataset_name: str):
    print(Path(PATH_TO_DATA, dataset_name))
    return Path(PATH_TO_DATA, dataset_name)
