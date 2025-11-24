from typing import Tuple
from pymonad.either import Either

from fedcore.api.api_configs import (
    LearningConfigTemplate, 
    LowRankTemplate
    )
from fedcore.algorithm.low_rank.low_rank_opt import LowRankModel
from fedcore.api.config_factory import ConfigFactory
from fedcore.api.utils.checkers_collection import DataCheck
from fedcore.architecture.dataset.api_loader import ApiLoader
from fedcore.architecture.utils.paths import  wrap_with_project_root_path
from fedcore.repository.constanst_repository import SLRStrategiesEnum
from fedot.core.data.data import InputData

def create_low_rank_with_prune_on_0_epoch(
          train_dataloader,
          val_dataloader,
          low_rank_treshold=0.05, 
          low_rank_mode=SLRStrategiesEnum._member_names_[0], 
          compose_mode=None, 
          epochs=1) -> Tuple[LowRankModel, InputData]:
    METRIC_TO_OPTIMISE = ['accuracy', 'latency']
    LOSS = 'cross_entropy'
    PEFT_PROBLEM = 'low_rank'
    INITIAL_MODEL_ASSUMPTION = 'ResNet18'
    PRETRAIN_SCENARIO = 'from_checkpoint'
    
    learning_template = LearningConfigTemplate(
        criterion=LOSS,
        learning_strategy=PRETRAIN_SCENARIO,
        peft_strategy=PEFT_PROBLEM,
        peft_strategy_params=LowRankTemplate(epochs=epochs),
    )


    al = ApiLoader('CIFAR10', {'split_ratio': [0.1, 0.9]})
    input_data = al._convert_to_fedcore(al._init_pretrain_dataset(al.source))
    input_data.train_dataloader = train_dataloader
    input_data.val_dataloader = val_dataloader
    
    LearningConfig = ConfigFactory.from_template(learning_template)
    learning_config = LearningConfig()
    print(learning_config)
    
    learning_config.update({'peft_strategy_params.strategy': low_rank_mode,
                                      'peft_strategy_params.non_adaptive_threshold': low_rank_treshold, 
                                      'peft_strategy_params.compose_mode': compose_mode,
                                      'peft_strategy_params.rank_prune_each': 0})


    data_cls = DataCheck(
        peft_task=learning_config.config['peft_strategy'],
        model={'path_to_model': wrap_with_project_root_path('pretrain_models/ResNet18_base.pth'),
                      'model_type': 'ResNet18'},
        learning_params=learning_config.learning_strategy_params
    )

    
    peft_params = learning_config.peft_strategy_params
    lowRankModel = LowRankModel(peft_params.to_dict())
    train_data = Either.insert(input_data).then(data_cls.check_input_data).value

    return lowRankModel, train_data