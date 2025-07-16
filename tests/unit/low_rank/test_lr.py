from copy import deepcopy
import pytest
from pymonad.either import Either
import torch
import torch.nn as nn
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
from fedcore.models.network_impl.decomposed_layers import DecomposedLinear


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

class MockModel(nn.Module):
    def __init__(self, input, hidden, output):
        super().__init__()
        
        self.dec_linear1 = DecomposedLinear(nn.Linear(input, hidden), decomposing_mode=True)
        self.dec_linear2 = DecomposedLinear(nn.Linear(hidden, output), decomposing_mode=True)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dec_linear1(x)
        x = self.relu(x)
        x = self.dec_linear2(x)
        return x

@pytest.fixture
def model():
    return MockModel(input=32, hidden=64, output=10)

@pytest.fixture
def base_params():
    return {
        'n_plateau': 3,
        'pl_thr': 1,
        'sv_thr': 0.01
    }

def test_trigger(model, base_params):
    pruner = DynamicRankPruner(base_params, model)
    
    pruner.traced_layers = {
        'layer1.S': [3, 3, 3],  
        'layer2.S': [3, 2, 1],  
    }
    pruner.traced_layers_ksis = {'layer1.S': 1.0, 'layer2.S': 1.0}
    
    result = pruner.trigger(epoch=0, kws={})
    
    assert isinstance(result, dict)
    assert 'dec_linear1.S' in result  
    assert 'dec_linear2.S' not in result 
    assert result['dec_linear1.S'] == 3  
    assert pruner.trigger_result == {'dec_linear1.S': 3}

def test_action(model, base_params):
    pruner = DynamicRankPruner(base_params, model)
    
    pruner.trigger_result = {
        'dec_linear1.S': 3,  
        'dec_linear2.S': 10  
    }
    
    pruner.action(epoch=0, kws={})
    
    assert model.dec_linear1._effective_rank == 3 
    assert model.dec_linear2._effective_rank == 10 
    assert 'dec_linear1.S' not in pruner.traced_layers 
    assert 'dec_linear2.S' not in pruner.traced_layers

# @pytest.mark.parametrize('rank_value,expected', [
#     (-2, 1),
#     (0, 1),
#     (1, 1),
#     (3, 3),
#     (5, 5),
#     (10, 5)  
# ])
@pytest.mark.parametrize('rank_value', [
    -2,    
    0,        
    1,  
    3,     
    32,     
    64,  
    10     
])
def test_action_edge_cases(rank_value, model, base_params):
    pruner = DynamicRankPruner(base_params, model)
    
    max_rank_layer1 = 32  
    max_rank_layer2 = 10  
    
    pruner.trigger_result = {
        'dec_linear1.S': rank_value,
        'dec_linear2.S': rank_value
    }
    
    pruner.action(epoch=0, kws={})
    
    expected_layer1 = 32
    expected_layer2 = 10
    
    assert model.dec_linear1._effective_rank == expected_layer1
    assert model.dec_linear2._effective_rank == expected_layer2