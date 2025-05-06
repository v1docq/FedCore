
from fedcore.api.api_configs import NeuralModelConfigTemplate, LearningConfigTemplate
from fedcore.api.config_factory import ConfigFactory
from fedcore.api.utils.checkers_collection import DataCheck
from fedcore.data.dataloader import load_data
from fedcore.models.data_operation.augmentation.pairwise_difference import PairwiseDifferenceModel

DATASET = 'CIFAR10'
INITIAL_MODEL = 'ResNet18'
LOSS = 'cross_entropy'
train_dataloader_params = {"batch_size": 64,
                           'shuffle': True,
                           'is_train': True,
                           'data_type': 'table',
                           'split_ratio': [0.01, 0.99]}
test_dataloader_params = {"batch_size": 100,
                          'shuffle': True,
                          'is_train': False,
                          'subset': 100,
                          'data_type': 'table'}


if __name__ == "__main__":


    pretrain_config = NeuralModelConfigTemplate(epochs=1,
                                                log_each=10,
                                                eval_each=15,
                                                save_each=50,
                                                criterion=LOSS,
                                                custom_learning_params=dict(use_early_stopping={'patience': 30,
                                                                                                'maximise_task': False,
                                                                                                'delta': 0.01}))
    learning_config = LearningConfigTemplate(criterion=LOSS,
                                             learning_strategy='from_scratch',
                                             learning_strategy_params=pretrain_config)
    pdl_config = ConfigFactory.from_template(learning_config)()
    fedcore_train_data = load_data(source=DATASET, loader_params=train_dataloader_params)
    fedcore_test_data = load_data(source=DATASET, loader_params=test_dataloader_params)
    input_train_data = DataCheck(model=INITIAL_MODEL, learning_params=None).init_input_data(fedcore_train_data)
    input_test_data = DataCheck(model=INITIAL_MODEL, learning_params=None).init_input_data(fedcore_test_data)
    pdl_model = PairwiseDifferenceModel(params=pdl_config)
    fitted = pdl_model.fit(input_train_data)
    predict = pdl_model.predict(input_test_data)
    _ = 1
