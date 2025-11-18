from typing import OrderedDict, Tuple
import pytest
from pymonad.either import Either
from fedcore.api.api_configs import (
    LearningConfigTemplate, 
    LowRankTemplate
    )
from fedcore.algorithm.low_rank.low_rank_opt import LowRankModel
from fedcore.api.config_factory import ConfigFactory
from fedcore.api.utils.checkers_collection import DataCheck
from fedcore.architecture.dataset.api_loader import ApiLoader
from fedcore.repository.constanst_repository import SLRStrategiesEnum
from fedcore.tools.dataload.small_cifar10_dataloader import get_small_cifar10_train_and_val_loaders


METRIC_TO_OPTIMISE = ['accuracy', 'latency']
LOSS = 'cross_entropy'
PEFT_PROBLEM = 'low_rank'
INITIAL_MODEL_ASSUMPTION = 'ResNet18'
PRETRAIN_SCENARIO = 'from_checkpoint'

def get_api_template() -> Tuple[type, OrderedDict]:
    learning_template = LearningConfigTemplate(
        criterion=LOSS,
        learning_strategy=PRETRAIN_SCENARIO,
        peft_strategy=PEFT_PROBLEM,
        peft_strategy_params=LowRankTemplate()
    )
    return learning_template

@pytest.mark.parametrize('low_rank_strategy', SLRStrategiesEnum._member_names_)
def test_lrs(low_rank_strategy):
    
    LearningConfig = ConfigFactory.from_template(get_api_template())
    learning_config = LearningConfig()
    learning_config.update({'peft_strategy_params.strategy': low_rank_strategy,
                                      'peft_strategy_params.non_adaptive_threshold': 0.05, 
                                      'peft_strategy_params.compose_mode': 'two_layers'})

    train_dataloader, val_dataloader = get_small_cifar10_train_and_val_loaders()
    al = ApiLoader('CIFAR10', {'split_ratio': [0.1, 0.9]})
    input_data = al._convert_to_fedcore(al._init_pretrain_dataset(al.source))
    input_data.train_dataloader = train_dataloader
    input_data.val_dataloader = val_dataloader
    data_cls = DataCheck(
        peft_task=learning_config.config['peft_strategy'],
        model=INITIAL_MODEL_ASSUMPTION,
        learning_params=learning_config.learning_strategy_params
    )
    train_data = Either.insert(input_data).then(data_cls.check_input_data).value


    peft_params = learning_config.peft_strategy_params
    lr = LowRankModel(peft_params.to_dict())

    lr_model = lr.fit(input_data=train_data)
