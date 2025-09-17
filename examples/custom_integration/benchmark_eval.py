import timeit

import numpy as np

from fedcore.data.custom.load_data import load_benchmark_data, split_benchmark_data
from fedcore.models.backbone.custom.boosting_model import FedcoreBoostingModel
from py_boost import GradientBoosting
from py_boost import SketchBoost
from fedcore.repository.data.custom.boosting_config import BOOSTING_MODEL_PARAMS


class ExperimentPipeline:
    def __init__(self):
        self.models_impl = dict(pyboost=GradientBoosting,
                                fedcore_boosting=FedcoreBoostingModel
                                )

    def init_boosting_model(self, model_name, model_params):
        self.model = self.models_impl[model_name](**model_params)

    def fit_boosting_model(self, dataset_dict: dict):
        eval_set = [{'X': dataset_dict['test_features'], 'y': dataset_dict['test_target']}]
        self.model.fit(dataset_dict['train_features'], dataset_dict['train_target'],
                       eval_sets=eval_set)

    def predict(self, dataset_dict):
        return self.model.predict(dataset_dict['test_features'])


def eval_boosting_perfomance(eval_func, X, X_test, y, y_test):
    start = timeit.default_timer()
    model = eval_func(X, X_test, y, y_test)
    end = timeit.default_timer()
    iter_per_sec = len(model.history) / (end - start)
    inference_list = []
    for i in range(10):
        start = timeit.default_timer()
        model.predict(X_test)
        end = timeit.default_timer()
        inference = end - start
        inference_list.append(inference)
    return dict(model=model, inference_time=np.mean(inference_list), learning_time=iter_per_sec)


def run_benchmark(dataset_id: int = 110, model_name: str = 'pyboost', model_params: dict = BOOSTING_MODEL_PARAMS):
    full_dataset_dict = load_benchmark_data(dataset_id)
    # apply randomization on initial dataset
    # sampled_dataset_dict = random_method(full_dataset_dict)
    train_dataset_dict = split_benchmark_data(full_dataset_dict)
    client = ExperimentPipeline()
    client.init_boosting_model(model_name, model_params)
    client.fit_boosting_model(train_dataset_dict)
    client.predict(train_dataset_dict)
    return

if __name__ == "__main__":
    run_benchmark()
