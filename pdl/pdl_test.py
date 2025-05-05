from enum import Enum

import openml
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score as accuracy, mean_squared_error as mse

from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.models.pdl.pairwise_model import PairwiseDifferenceClassifier


def get_data_openml(idx: int, target: str, task='classification', head=None):
    openml.datasets.list_datasets(output_format="dataframe")
    dataset = openml.datasets.get_dataset(idx)
    data = dataset.get_data(dataset_format="dataframe")
    df = data[0]

    df = shuffle(df)
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent').set_output(transform='pandas')
    le = LabelEncoder()
    if head is not None:
        df = df.head(head)

    # Сюда также надо добавить лейбл энкодинг
    X, y = df.drop(columns=[target]), df[target]
    X = imputer.fit_transform(X)

    for column in X.columns:
        if X[column].dtype == 'object':
            X[column] = le.fit_transform(X[column].astype(str))
    if task == 'classification':
        task = Task(TaskTypesEnum.classification)
    else:
        task = Task(TaskTypesEnum.regression)

    return InputData(
        idx=np.arange(0, len(X)),
        features=X,
        target=y.values,
        data_type=DataTypesEnum.table,
        task=task
    )


class Datasets(Enum):
    UNDER_30_DATA_BINARY = dict(
        shuttle_15x7=(get_data_openml(idx=172, target='Class'), 'binary'),
        trains_10x33=(get_data_openml(idx=52, target='class'), 'binary'),
        analcat_31x16=(get_data_openml(idx=760, target='binaryClass'), 'binary'),
    )
    UNDER_30_DATA_MULTI = dict(
        lung_32x57=(get_data_openml(idx=163, target='class'), 'multi'),
        pasture_36x23=(get_data_openml(idx=339, target='pasture-prod-class'), 'multi'),
    )
    UNDER_30_DATA_REG = dict(
        cristalli_32x1k=(get_data_openml(idx=420, target='oz1143', task='reg'), 'reg'),
        longley_16x7=(get_data_openml(idx=211, target='employed', task='reg'), 'reg'),
        detroit_13x14=(get_data_openml(idx=208, target='ASR', task='reg'), 'reg'),
    )

    UNDER_60_DATA_BINARY = dict(
        aids_50x5=(get_data_openml(idx=346, target='Sex'), 'binary'),
        diabetes_43x3=(get_data_openml(idx=791, target='binaryClass'), 'binary'),
        labor_57x17=(get_data_openml(idx=4, target='class'), 'binary'),
    )
    UNDER_60_DATA_MULTI = dict(
        squash_52x24=(get_data_openml(idx=342, target='Acceptability'), 'multi'),
        clover_63x32=(get_data_openml(idx=343, target='WhiteClover-94'), 'multi'),
    )
    UNDER_60_DATA_REG = dict(
        sleuth_50x7=(get_data_openml(idx=707, target='rank', task='reg'), 'reg'),
        mbagrade_61x3=(get_data_openml(idx=190, target='grade_point_average', task='reg'), 'reg'),
        elusage_55x3=(get_data_openml(idx=228, target='average_electricity_usage', task='reg'), 'reg'),
    )

    UNDER_100_DATA_BINARY = dict(
        fri_100x6=(get_data_openml(idx=754, target='binaryClass'), 'binary'),
        cloud_108x8=(get_data_openml(idx=890, target='binaryClass'), 'binary'),
        ar4_107x30=(get_data_openml(idx=1061, target='defects'), 'binary'),
    )
    UNDER_100_DATA_MULTI = dict(
        postoperative_90x9=(get_data_openml(idx=34, target='decision'), 'multi'),
        grub_155x9=(get_data_openml(idx=338, target='GG_new'), 'multi'),
    )
    UNDER_100_DATA_REG = dict(
        baskball_96x5=(get_data_openml(idx=214, target='points_per_minute', task='reg'), 'reg'),
        nasa_93x24=(get_data_openml(idx=1076, target='act_effort', task='reg'), 'reg'),
    )


class Estimator:
    def __init__(self, model):
        self.model = model
        under_30_data = [
            Datasets.UNDER_30_DATA_BINARY,
            Datasets.UNDER_30_DATA_MULTI,
            Datasets.UNDER_30_DATA_REG
        ]
        under_60_data = [
            Datasets.UNDER_60_DATA_BINARY,
            Datasets.UNDER_60_DATA_MULTI,
            Datasets.UNDER_60_DATA_REG
        ]
        under_100_data = [
            Datasets.UNDER_100_DATA_BINARY,
            Datasets.UNDER_100_DATA_MULTI,
            Datasets.UNDER_100_DATA_REG
        ]
        self.datasets = [under_30_data, under_60_data, under_100_data]

    def fit_predict(self):
        model_score_array = []
        # (under_30, under_60, under_100)
        for dataset_group in self.datasets:
            # (BINARY, MULTI, REG)
            for dataset_type in dataset_group:
                datasets_dict = dataset_type.value
                for data_name, (data, task_type) in datasets_dict.items():
                    print('=' * 50)
                    print(f'Current dataset: {data_name}. Task - {task_type}')

                    # train/test split
                    train, test = train_test_data_setup(data)

                    # fit & predict
                    self.model.fit(train)
                    preds = self.model.predict(test)

                    # get score
                    if task_type == 'reg':
                        score = mse(test.target, preds.predict)
                        print(f'MSE: {score:.4f}')
                    else:
                        score = accuracy(test.target, preds.predict)
                        print(f'Accuracy: {score:.4f}')

                    model_score_array.append((data_name, score))

        return model_score_array


params = OperationParameters(model='rf', n_estimators=10)
pd_clf = PairwiseDifferenceClassifier(params)

est_pd = Estimator(pd_clf)
model_score_array = est_pd.fit_predict()


