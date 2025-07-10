from copy import deepcopy
import pytest
from pymonad.either import Either
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from fedcore.api.api_configs import (
    APIConfigTemplate, 
    AutoMLConfigTemplate, 
    FedotConfigTemplate, 
    LearningConfigTemplate, 
    ModelArchitectureConfigTemplate, 
    NeuralModelConfigTemplate, 
    LowRankTemplate
    )
from fedcore.algorithm.low_rank.low_rank_opt import LowRankModel
from fedcore.api.config_factory import ConfigFactory
from fedcore.api.utils.checkers_collection import DataCheck
from fedcore.architecture.dataset.api_loader import ApiLoader
from fedcore.tools.example_utils import get_scenario_for_api
from fedcore.repository.constanst_repository import SLRStrategiesEnum
from fedcore.algorithm.low_rank.hooks import DynamicRankPruner
from fedcore.architecture.abstraction.accessor import Accessor


METRIC_TO_OPTIMISE = ['accuracy', 'latency']
LOSS = 'cross_entropy'
PEFT_PROBLEM = 'low_rank'
INITIAL_ASSUMPTION = 'ResNet18'
PRETRAIN_SCENARIO = 'from_checkpoint'

def get_api_template():
    learning_template = LearningConfigTemplate(
        criterion=LOSS,
        learning_strategy=PRETRAIN_SCENARIO,
        peft_strategy=PEFT_PROBLEM,
        peft_strategy_params=LowRankTemplate()
    )
    return learning_template


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

@pytest.mark.parametrize('low_rank_strategy', SLRStrategiesEnum._member_names_)
def test_lrs(low_rank_strategy):
    
    LearningConfig = ConfigFactory.from_template(get_api_template())
    learning_config = LearningConfig({'peft_strategy_params.low_rank_strategy': low_rank_strategy,
                                      'peft_strategy_params.non_adaptive_threshold': 0.05, 
                                      'peft_strategy_params.compose_mode': 'two_layers'})

    train_dataloader, val_dataloader = get_small_dataset()
    al = ApiLoader('CIFAR10', {'split_ratio': [0.1, 0.9]})
    input_data = al._convert_to_fedcore(al._init_pretrain_dataset(al.source))
    input_data.train_dataloader = train_dataloader
    input_data.val_dataloader = val_dataloader
    data_cls = DataCheck(
        peft_task=learning_config.config['peft_strategy'],
        model=INITIAL_ASSUMPTION,
        learning_params=learning_config.learning_strategy_params
    )
    train_data = Either.insert(input_data).then(data_cls.check_input_data).value


    peft_params = learning_config.peft_strategy_params
    lr = LowRankModel(peft_params.to_dict())

    lr_model = lr.fit(input_data=train_data)

    pruner_params = {
        'n_plateau': 3,
        'pl_thr': 0.03,
        'sv_thr': 1e-5
    }
    
    pruner = DynamicRankPruner(
        params=pruner_params,
        model=lr_model
    )
    
    assert hasattr(pruner, 'traced_layers')
    assert len(pruner.traced_layers) > 0

    test_input = torch.randn(1, 3, 32, 32)  
    history_ranks = [50.3, 50.1, 50.05, 50.01, 50.005] 
    
    for epoch, rank in enumerate(history_ranks, 1):
        for name in pruner.traced_layers:
            layer = Accessor.get_module(lr_model, name)
            with torch.no_grad():
                layer.S.fill_(rank)
        
        pruner._update_stable_ranks()
        
        if epoch >= pruner.n_plateau:
            to_prune = pruner.trigger(epoch, {})
            if epoch == len(history_ranks):
                assert len(to_prune) > 0 
                pruner.action(epoch, {})
                
                for name in to_prune:
                    layer = Accessor.get_module(lr_model, name)
                    assert hasattr(layer, '_effective_rank')
                     