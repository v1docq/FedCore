from copy import deepcopy
import pytest
import torch
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
    QuantTemplate
    )
from fedcore.algorithm.quantization.quantizers import BaseQuantizer
from fedcore.api.config_factory import ConfigFactory
from fedcore.api.utils.checkers_collection import DataCheck
from fedcore.architecture.dataset.api_loader import ApiLoader
from fedcore.tools.example_utils import get_scenario_for_api
from fedcore.architecture.computational.devices import default_device


def get_api_template(quant_type: str):
    METRIC_TO_OPTIMISE = ['accuracy', 'latency']
    LOSS = 'cross_entropy'
    PROBLEM = 'classification'
    PEFT_PROBLEM = 'quantization'
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

    qat_config = NeuralModelConfigTemplate(epochs=3,
                                            log_each=3,
                                            eval_each=3,
                                            criterion=LOSS,
                                            )
    
    peft_config = QuantTemplate(quant_type=quant_type,
                                allow_emb=False,
                                allow_conv=True,
                                qat_params=qat_config
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


def get_reduction(model_before, model_after):
    
    params_before = sum(p.numel() for p in model_before.parameters() if p.requires_grad)
    params_after = sum(p.numel() for p in model_after.parameters() if p.requires_grad)
    return params_after / params_before

@pytest.mark.parametrize('quant_type', ['dynamic', 'static', 'qat'])
def test_quantizers(quant_type):
    api = get_api_template(quant_type=quant_type)
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

    quant_params = api_config.learning_config.peft_strategy_params
    quantizer = BaseQuantizer(quant_params.to_dict())

    quant_model = quantizer.fit(input_data=train_data)
    assert quant_model is not None
    params_dict = quantizer.estimate_params(example_batch=quantizer.data_batch_for_calib,
                                         model_before=quantizer.model_before_quant,
                                         model_after=quantizer.model_after_quant)
    assert params_dict is not None
    reduction = get_reduction(model_after=quant_model, model_before=quantizer.model_before_quant)
    if default_device() == torch.device('cpu'):
        assert reduction < 1, f'{quant_type} quantization doesnt reduce number of model parameters, reduction: {reduction}'
    else:
        assert reduction <= 1