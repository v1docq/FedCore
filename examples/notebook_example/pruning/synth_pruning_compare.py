import os
from copy import deepcopy
from pymonad.either import Either

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from fedcore.api.api_configs import (
    APIConfigTemplate,
    AutoMLConfigTemplate,
    FedotConfigTemplate,
    LearningConfigTemplate,
    ModelArchitectureConfigTemplate,
    TrainingTemplate,
    PruningTemplate
)
from fedcore.algorithm.pruning.pruners import BasePruner
from fedcore.api.config_factory import ConfigFactory
from fedcore.api.utils.checkers_collection import DataCheck
from fedcore.architecture.dataset.api_loader import ApiLoader
from fedcore.tools.example_utils import get_scenario_for_api
from fedcore.architecture.utils.paths import PROJECT_PATH


def get_api_template(pruner_importance: str):
    METRIC_TO_OPTIMISE = ['accuracy', 'latency']
    LOSS = 'cross_entropy'
    PROBLEM = 'classification'
    PEFT_PROBLEM = 'pruning'
    path_to_model = 'examples/api_example/pruning/cv_task/pretrain_models/pretrain_model_checkpoint_at_15_epoch.pt'
    path = os.path.join(PROJECT_PATH, path_to_model)
    INITIAL_ASSUMPTION = {
        'path_to_model': path,
        'model_type': 'ResNet18'}
    PRETRAIN_SCENARIO = 'from_checkpoint'
    POP_SIZE = 1

    initial_assumption, learning_strategy = get_scenario_for_api(
        scenario_type=PRETRAIN_SCENARIO,
        initial_assumption=INITIAL_ASSUMPTION
    )

    model_config = ModelArchitectureConfigTemplate(
        input_dim=None,
        output_dim=None,
        depth=6
    )

    pretrain_config = TrainingTemplate(
        epochs=200,
        log_each=10,
        eval_each=15,
        save_each=50,
        criterion=LOSS,
        model_architecture=model_config,
        custom_learning_params=dict(use_early_stopping={'patience': 30,
                                                        'maximise_task': False,
                                                        'delta': 0.01})
    )

    fedot_config = FedotConfigTemplate(
        problem=PROBLEM,
        metric=METRIC_TO_OPTIMISE,
        pop_size=POP_SIZE,
        timeout=5,
        initial_assumption=initial_assumption
    )

    automl_config = AutoMLConfigTemplate(fedot_config=fedot_config)

    finetune_config = TrainingTemplate(
        epochs=3,
        log_each=3,
        eval_each=3,
        criterion=LOSS
    )
    peft_config = PruningTemplate(
        importance=pruner_importance,
        # importance="magnitude",
        pruning_ratio=0.8,
        finetune_params=finetune_config
    )

    learning_config = LearningConfigTemplate(
        criterion=LOSS,
        learning_strategy=learning_strategy,
        learning_strategy_params=pretrain_config,
        peft_strategy=PEFT_PROBLEM,
        peft_strategy_params=peft_config
    )

    api_template = APIConfigTemplate(
        automl_config=automl_config,
        learning_config=learning_config
    )

    return api_template


def get_small_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    small_train_subset = Subset(dataset, range(100))
    small_val_subset = Subset(dataset, range(100, 200))

    train_dataloader = DataLoader(small_train_subset, batch_size=10, shuffle=True)
    val_dataloader = DataLoader(small_val_subset, batch_size=10, shuffle=False)

    return train_dataloader, val_dataloader


def test_pruners(pruner_name):
    api = get_api_template(pruner_importance=pruner_name)
    APIConfig = ConfigFactory.from_template(api)
    api_config = APIConfig()

    train_dataloader, val_dataloader = get_small_dataset()
    al = ApiLoader('CIFAR10', {'split_ratio': [0.1, 0.9]})
    input_data = al._convert_to_fedcore(al._init_pretrain_dataset(al.source))
    input_data.train_dataloader = train_dataloader
    input_data.val_dataloader = val_dataloader
    data_cls = DataCheck(
        peft_task=api_config.learning_config.config['peft_strategy'],
        model=api_config.automl_config.fedot_config['initial_assumption'],
        learning_params=api_config.learning_config.learning_strategy_params
    )
    train_data = Either.insert(input_data).then(deepcopy).then(data_cls.check_input_data).value

    pruning_params = api_config.learning_config.peft_strategy_params
    pruner = BasePruner(pruning_params.to_dict())

    pruned_model = pruner.fit(input_data=train_data, finetune=False)
    params_dict = pruner.estimate_params(example_batch=pruner.data_batch_for_calib,
                                         model_before=pruner.model_before,
                                         model_after=pruner.model_after)
    if params_dict['params_before'] == params_dict['params_after']:
        print(f'Pruning operator - {pruner_name} working incorrect')
        pruned_model = pruner.fit(input_data=train_data, finetune=False)
        params_dict = pruner.estimate_params(example_batch=pruner.data_batch_for_calib,
                                             model_before=pruner.model_before,
                                             model_after=pruner.model_after)
    return params_dict

pruner_with_reg = ['bn_scale','group_magnitude', 'group_taylor', 'group_hessian']
pruner_with_grad = ['hessian', 'taylor']
zeroshot_pruner = ['magnitude', 'lamp', 'random']
all_pruners = zeroshot_pruner + pruner_with_grad + pruner_with_grad
if __name__ == "__main__":
    for pruner_name in pruner_with_reg:
        params_dict = test_pruners(pruner_name)
