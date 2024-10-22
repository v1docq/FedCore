import ssl

import dask
import distributed.dashboard.components.scheduler as dashboard
from distributed import Client, LocalCluster
from distributed.security import Security
from weakref import WeakValueDictionary
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedcore.architecture.preprocessing.data_convertor import (
    CustomDatasetCLF,
    CustomDatasetTS,
    DataConverter,
    TensorConverter,
)
from fedcore.architecture.settings.computational import backend_methods as np


# from dask.distributed import LocalCluster, Client


def fedot_data_type(func):
    def decorated_func(self, *args):
        if not isinstance(args[0], InputData):
            args[0] = DataConverter(data=args[0])
        features = args[0].features

        if len(features.shape) < 4:
            try:
                input_data_squeezed = np.squeeze(features, 3)
            except ValueError:
                input_data_squeezed = np.squeeze(features)
        else:
            input_data_squeezed = features
        return func(self, input_data_squeezed, args[1])

    return decorated_func


def convert_to_4d_torch_array(func):
    def decorated_func(self, *args):
        init_data = args[0]
        data = DataConverter(data=init_data).convert_to_4d_torch_format()
        if isinstance(init_data, InputData):
            init_data.features = data
        else:
            init_data = data
        return func(self, init_data)

    return decorated_func


def convert_to_3d_torch_array(func):
    def decorated_func(self, *args):
        init_data = args[0]
        data = DataConverter(data=init_data).convert_to_torch_format()
        if isinstance(init_data, InputData):
            init_data.features = data
        else:
            init_data = data
        return func(self, init_data, args[1])

    return decorated_func


def convert_inputdata_to_torch_dataset(func):
    def decorated_func(self, *args):
        ts = args[0]
        return func(self, CustomDatasetCLF(ts))

    return decorated_func


def convert_inputdata_to_torch_time_series_dataset(func):
    def decorated_func(self, *args):
        ts = args[0]
        return func(self, CustomDatasetTS(ts))

    return decorated_func


def convert_to_torch_tensor(func):
    def decorated_func(self, *args):
        data = TensorConverter(data=args[0]).convert_to_tensor(data=args[0])
        return func(self, data)

    return decorated_func


def remove_1_dim_axis(func):
    def decorated_func(self, *args):
        time_series = np.nan_to_num(args[0])
        if any([dim == 1 for dim in time_series.shape]):
            time_series = DataConverter(data=time_series).convert_to_1d_array()
        return func(self, time_series)

    return decorated_func


def convert_to_input_data(func):
    def decorated_func(*args, **kwargs):
        features, names = func(*args, **kwargs)
        ts_data = InputData(
            idx=np.arange(len(features)),
            features=features,
            target="no_target",
            task="no_task",
            data_type=DataTypesEnum.table,
            supplementary_data={"feature_name": names},
        )
        return ts_data

    return decorated_func


class Singleton(type):
    _instances = WeakValueDictionary()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super(Singleton, cls).__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class DaskServer(metaclass=Singleton):
    def __init__(self):
        self._overload_dask_config()
        print("Creating Dask Server")
        cluster = LocalCluster(processes=False, security=self.sec)
        # connect client to your cluster
        self.client = Client(cluster)

    def _overload_dask_config(self):
        self.sec = Security(
            tls_max_version=ssl.TLSVersion.TLSv1_3,
            tls_min_version=ssl.TLSVersion.TLSv1_2,
        )
        dask.config.set({"distributed.scheduler.idle-timeout": "5 minutes"})
        # Shut down the scheduler after this duration if no activity has occurred
        dask.config.set({"distributed.scheduler.no-workers-timeout": "5 minutes"})
        # Timeout for tasks in an unrunnable state. If task remains unrunnable for longer than this, it fails.
        # A task is considered unrunnable IFF it has no pending dependencies,
        # and the task has restrictions that are not satisfied by any available worker
        # or no workers are running at all. In adaptive clusters,
        # this timeout must be set to be safely higher than the time it takes for workers to spin up.
        dask.config.set({"distributed.worker.lifetime.duration": "1 hour"})
        # The time after creation to close the worker, like "1 hour"
        setattr(dashboard, "BOKEH_THEME", "night_sky")
        # 'caliber'
        # 'light_minimal'
        # 'dark_minimal'
        # 'night_sky'
        # 'contrast'
