from copy import deepcopy
import pytest
from pymonad.either import Either

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from fedcore.api.api_configs import (
    APIConfigTemplate, 
    AutoMLConfigTemplate, 
    FedotConfigTemplate, 
    LearningConfigTemplate, 
    ModelArchitectureConfigTemplate, 
    NeuralModelConfigTemplate, 
    PruningTemplate
    )
from fedcore.algorithm.pruning.pruners import BasePruner
from fedcore.api.config_factory import ConfigFactory
from fedcore.api.utils.checkers_collection import DataCheck
from fedcore.architecture.dataset.api_loader import ApiLoader
from fedcore.tools.example_utils import get_scenario_for_api


def get_api_template(pruner_importance: str):
    METRIC_TO_OPTIMISE = ['accuracy', 'latency']
    LOSS = 'cross_entropy'
    PROBLEM = 'classification'
    PEFT_PROBLEM = 'pruning'
    INITIAL_ASSUMPTION = 'ResNet18'
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


    pretrain_config = NeuralModelConfigTemplate(
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

    finetune_config = NeuralModelConfigTemplate(
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


@pytest.mark.parametrize('pruner_name', ['magnitude', 'hessian', 'bn_scale', 
                                         'lamp', 'random', 'group_magnitude', 
                                         'group_taylor', 'group_hessian', 'taylor', ])
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
                                         model_before=pruner.model_before_pruning,
                                         model_after=pruner.model_after_pruning)

    assert f'{pruner_name} pruner doesnt reduce number of model parameters'
