import logging
import os
import shutil
import urllib.request as request
import zipfile
from pathlib import Path
from typing import Optional, Union
import numpy as np
import chardet
import pandas as pd
from datasets import load_dataset
from datasetsforecast.m3 import M3
from datasetsforecast.m4 import M4
from datasetsforecast.m5 import M5

from fedcore.architecture.dataset.utils import read_tsv_or_csv, read_txt_files, read_ts_files, read_arff_files


class TimeSeriesLoader:
    """Class for reading data time series files.
    At the moment supports ``.ts``, ``.txt``, ``.tsv``, and ``.arff`` formats.

    Args:
        folder: path to folder with data

    Examples:
        >>> data_loader = TimeSeriesLoader()
        >>> train_data, test_data = data_loader.load_data('./ts_file.tsv')
    """

    def load_data(self, path_to_data: str):
        is_tsv_or_csv_file = path_to_data.__contains__('.tsv') or path_to_data.__contains__('.csv')
        is_txt_file = path_to_data.__contains__('.txt')
        is_ts_file = path_to_data.__contains__('.ts')
        is_arff_file = path_to_data.__contains__('.arff')

        if is_tsv_or_csv_file:
            mode = 'tsv' if path_to_data.__contains__('.tsv') else 'csv'
            features, target = read_tsv_or_csv(path_to_data, mode=mode)
            features = features.values
        elif is_txt_file:
            xfeatures, target = read_txt_files(path_to_data)
        elif is_ts_file:
            features, target = read_ts_files(path_to_data)
        elif is_arff_file:
            features, target = read_arff_files(path_to_data)

        return features, target


class DataLoader:
    """Class for reading data files and downloading from UCR archive if not found locally.
    At the moment supports ``.ts``, ``.txt``, ``.tsv``, and ``.arff`` formats.

    Args:
        dataset_name: name of dataset
        folder: path to folder with data

    Examples:
        >>> data_loader = DataLoader('ItalyPowerDemand')
        >>> train_data, test_data = data_loader.load_data()
    """

    def __init__(self, dataset_name: str, folder: Optional[str] = None, source_url: Optional[str] = None):
        self.logger = logging.getLogger('DataLoader')
        self.url = source_url if source_url is not None else f'http://www.timeseriesclassification.com/aeon-toolkit/'
        self.dataset_name = dataset_name
        self.folder = folder
        self.forecast_data_source = {
            'M3': M3.load,
            'M4': M4.load,
            # 'M4': self.local_m4_load,
            'M5': M5.load,
            'monash_tsf': load_dataset
        }

    def load_forecast_data(self, forecast_family: Optional[str] = None, folder: Optional[Union[Path, str]] = None):
        if forecast_family not in self.forecast_data_source:
            forecast_family = self.dataset_name.get('benchmark') if isinstance(self.dataset_name, dict) else 'M4'
        if folder is None:
            folder = EXAMPLES_DATA_PATH
        loader = self.forecast_data_source[forecast_family]
        dataset_name = self.dataset_name.get('dataset') if isinstance(self.dataset_name, dict) else self.dataset_name
        group_df, _, _ = loader(directory=folder, group=f'{M4_PREFIX[dataset_name[0]]}')
        ts_df = group_df[group_df['unique_id'] == dataset_name]
        del ts_df['unique_id']
        ts_df = ts_df.set_index('datetime') if 'datetime' in ts_df.columns else ts_df.set_index('ds')
        train_data = ts_df.values.flatten()
        target = train_data[-self.dataset_name['task_params']['forecast_length']:].flatten()
        train_data = (train_data, target)
        return train_data, train_data

    @staticmethod
    def local_m4_load(group: Optional[str] = None):
        path_to_result = EXAMPLES_DATA_PATH + '/forecasting/'
        for result_cvs in os.listdir(path_to_result):
            if result_cvs.__contains__(group):
                return pd.read_csv(Path(path_to_result, result_cvs))

    def load_data(self, shuffle: bool = True) -> tuple:
        """Load data for classification experiment locally or externally from UCR archive.

        Returns:
            tuple: train and test data
        """
        dataset_name = self.dataset_name
        data_path = os.path.join(PROJECT_PATH, 'fedot_ind', 'data') if self.folder is None else self.folder
        _, train_data, test_data = self.read_train_test_files(dataset_name=dataset_name,
                                                              data_path=data_path,
                                                              shuffle=shuffle)
        if train_data is None:
            self.logger.info(f'Downloading {dataset_name} from {self.url}...')

            # Create temporary folder for downloaded data
            cache_path = os.path.join(PROJECT_PATH, 'temp_cache/')
            download_path = cache_path + 'downloads/'
            temp_data_path = cache_path + 'temp_data/'
            for _ in (download_path, temp_data_path):
                os.makedirs(_, exist_ok=True)

            url = self.url + f'/{dataset_name}.zip'
            request.urlretrieve(url, download_path + f'temp_data_{dataset_name}')
            try:
                zipfile.ZipFile(download_path + f'temp_data_{dataset_name}').extractall(temp_data_path + dataset_name)
            except zipfile.BadZipFile:
                raise FileNotFoundError(f'Cannot extract data: {dataset_name} dataset not found in {self.url}')
            else:
                self.logger.info(f'{dataset_name} data downloaded. Unpacking...')
                train_data, test_data = self.extract_data(dataset_name, temp_data_path)
                shutil.rmtree(cache_path)

        self.logger.info('Data read successfully from local folder')

        if isinstance(train_data[0].iloc[0, 0], pd.Series):
            def convert(arr):
                """Transform pd.Series values to np.ndarray"""
                return np.array([d.values for d in arr])

            train_data = (np.apply_along_axis(convert, 1, train_data[0]), train_data[1])
            test_data = (np.apply_along_axis(convert, 1, test_data[0]), test_data[1])

        return train_data, test_data

    def read_train_test_files(self, data_path: Union[Path, str], dataset_name: str, shuffle: bool = True):

        dataset_dir_path = os.path.join(data_path, dataset_name)
        file_path = dataset_dir_path + f'/{dataset_name}_TRAIN'
        is_multivariate = False
        self.logger.info(f'Reading data from {dataset_dir_path}')

        if os.path.isfile(file_path + '.tsv'):
            x_train, y_train, x_test, y_test = self.read_tsv_or_csv(dataset_name, data_path, mode='tsv')
        elif os.path.isfile(file_path + '.txt'):
            x_train, y_train, x_test, y_test = self.read_txt_files(dataset_name, data_path)
        elif os.path.isfile(file_path + '.ts'):
            x_train, y_train, x_test, y_test = self.read_ts_files(dataset_name, data_path)
            is_multivariate = True
        elif os.path.isfile(file_path + '.arff'):
            x_train, y_train, x_test, y_test = self.read_arff_files(dataset_name, data_path)
            is_multivariate = True
        elif os.path.isfile(file_path + '.csv'):
            x_train, y_train, x_test, y_test = self.read_tsv_or_csv(dataset_name, data_path, mode='csv')
        else:
            self.logger.error(f'Data not found in {dataset_dir_path}')
            return None, None, None

        y_train, y_test = convert_type(y_train, y_test)

        if shuffle:
            shuffled_idx = np.arange(x_train.shape[0])
            np.random.shuffle(shuffled_idx)
            if isinstance(x_train, pd.DataFrame):
                x_train = x_train.iloc[shuffled_idx, :]
            else:
                x_train = x_train[shuffled_idx, :]
            y_train = y_train[shuffled_idx]
        return is_multivariate, (x_train, y_train), (x_test, y_test)

    @staticmethod
    def predict_encoding(file_path: Union[Path, str], n_lines: int = 20) -> str:
        with Path(file_path).open('rb') as f:
            rawdata = b''.join([f.readline() for _ in range(n_lines)])
        return chardet.detect(rawdata)['encoding']

    def extract_data(self, dataset_name: str, data_path: str):
        """Unpacks data from downloaded file and saves it into Data folder with ``.tsv`` extension.

        Args:
            dataset_name: name of dataset
            data_path: path to folder downloaded data

        Returns:
            tuple: train and test data

        """
        try:
            is_multi, (x_train, y_train), (x_test, y_test) = self.read_train_test_files(
                data_path, dataset_name)

        except Exception as e:
            self.logger.error(f'Error while unpacking data: {e}')
            return None, None

        # Conversion of target values to int or str
        y_train, y_test = convert_type(y_train, y_test)

        # Save data to tsv files
        new_path = os.path.join(PROJECT_PATH, 'fedot_ind', 'data') if self.folder is None else self.folder
        new_path = os.path.join(new_path, dataset_name)
        os.makedirs(new_path, exist_ok=True)

        self.logger.info(f'Saving {dataset_name} data files to {new_path}')
        for subset in ('TRAIN', 'TEST'):
            if not is_multi:
                df = pd.DataFrame(x_train if subset == 'TRAIN' else x_test)
                df.insert(0, 'class', y_train if subset == 'TRAIN' else y_test)
                df.to_csv(
                    os.path.join(
                        new_path,
                        f'{dataset_name}_{subset}.tsv'),
                    sep='\t',
                    index=False,
                    header=False)
                del df

            else:
                old_path = os.path.join(
                    data_path, dataset_name, f'{dataset_name}_{subset}.ts')
                shutil.move(old_path, new_path)

        if is_multi:
            return (x_train, y_train), (x_test, y_test)
        else:
            return (pd.DataFrame(x_train),
                    y_train), (pd.DataFrame(x_test), y_test)


def convert_type(y_train, y_test):
    # Conversion of target values to int or str
    try:
        y_train = y_train.astype('float')
        y_test = y_test.astype('float')
    except ValueError:
        y_train = y_train.astype(str)
        y_test = y_test.astype(str)
    return y_train, y_test
