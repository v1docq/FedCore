import os

from fedcore.api.api_configs import NeuralModelConfigTemplate, LearningConfigTemplate, ModelArchitectureConfigTemplate
from fedcore.api.config_factory import ConfigFactory
from fedcore.api.utils.checkers_collection import DataCheck
from fedcore.architecture.utils.paths import PATH_TO_DATA
from fedcore.data.dataloader import load_data
from fedcore.models.data_operation.augmentation.pairwise_difference import PairwiseDifferenceModel
from fedcore.repository.constanst_repository import TS_MULTI_CLF_BENCHMARK
import torch
from sklearn.metrics import f1_score, accuracy_score

# DATASET = 'CIFAR10'
INITIAL_MODEL = 'InceptionNet'
model_config = ModelArchitectureConfigTemplate(input_dim=None,
                                               output_dim=None,
                                               depth=6)
LOSS = 'cross_entropy'
train_dataloader_params = {"batch_size": 10,
                           'shuffle': True,
                           'is_train': True,
                           'data_type': 'time_series',
                           'split_ratio': [0.9, 0.1]}
test_dataloader_params = {"batch_size": 10,
                          'shuffle': False,
                          'is_train': False,
                          # 'subset': 100,
                          'data_type': 'time_series'}


def load_benchmark_data(dataset_name, backbone_model, train_dataloader_params, test_dataloader_params,
                        model_params: dict = None):
    if isinstance(dataset_name, dict):
        fedcore_train_data = load_data(source=dataset_name['train_path'], loader_params=train_dataloader_params)
        fedcore_test_data = load_data(source=dataset_name['test_path'], loader_params=test_dataloader_params)
    else:
        fedcore_train_data = load_data(source=dataset_name, loader_params=train_dataloader_params)
        fedcore_test_data = load_data(source=dataset_name, loader_params=test_dataloader_params)
    input_train_data = DataCheck(model=backbone_model, learning_params=model_params).init_input_data(fedcore_train_data)
    input_test_data = DataCheck(model=backbone_model, learning_params=model_params).init_input_data(fedcore_test_data)
    return input_train_data, input_test_data


def build_config(model_config):
    pretrain_config = NeuralModelConfigTemplate(epochs=100,
                                                log_each=10,
                                                eval_each=200,
                                                save_each=200,
                                                criterion=LOSS,
                                                model_architecture=model_config,
                                                custom_learning_params=dict(use_early_stopping={'patience': 30,
                                                                                                'maximise_task': False,
                                                                                                'delta': 0.01}))
    learning_config = LearningConfigTemplate(criterion=LOSS,
                                             learning_strategy='from_scratch',
                                             learning_strategy_params=pretrain_config)
    model_learning_config = ConfigFactory.from_template(learning_config)()
    pretrain_config = ConfigFactory.from_template(pretrain_config)()
    return model_learning_config, pretrain_config


def eval_score(input_test, predict):
    target = list(input_test.features.train_dataloader)
    pred = predict.cpu().detach().numpy()
    target = torch.concat([x[1] for x in target])
    f1 = f1_score(target, pred, average='weighted')
    acc = accuracy_score(target, pred)
    return (f1, acc)


if __name__ == "__main__":
    dict_result = {}
    for dataset_name in TS_MULTI_CLF_BENCHMARK:
        PATH_TO_TRAIN = os.path.join(PATH_TO_DATA, 'time_series_classification', 'multi_dim', dataset_name,
                                     f'{dataset_name}_TRAIN.ts')
        PATH_TO_TEST = os.path.join(PATH_TO_DATA, 'time_series_classification', 'multi_dim', dataset_name,
                                    f'{dataset_name}_TEST.ts')
        if not os.path.exists(PATH_TO_TRAIN):
            break
        dataset_path = {'train_path': PATH_TO_TRAIN,
                        'test_path': PATH_TO_TEST}
        model_learning_config, pretrain_config = build_config(model_config)
        input_train, input_test = load_benchmark_data(dataset_path, INITIAL_MODEL,
                                                      train_dataloader_params, test_dataloader_params,
                                                      pretrain_config)

        pdl_model = PairwiseDifferenceModel(params=model_learning_config)
        fitted = pdl_model.fit(input_train)
        predict = pdl_model.predict(input_test)
        score = eval_score(input_test, predict)
        dict_result.update({dataset_name: score})
        print(dict_result)
    _ = 1
