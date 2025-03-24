from fedcore.tools.example_utils import get_scenario_for_api
from fedcore.api.main import FedCore
from fedcore.api.utils.checkers_collection import ApiConfigCheck
from fedcore.data.dataloader import load_data
from fedcore.repository.config_repository import DEFAULT_CLF_API_CONFIG
from torchvision.models import resnet18
import torch

model = resnet18(pretrained=True)
torch.save(model.state_dict(), './pretrain_model/resnet_base_pretrain.pt')

DATASET = 'CIFAR10'
DATASET_PARAMS = {'train_bs': 64,
                  'val_bs': 100,
                  'train_shuffle': True,
                  'val_shuffle': False}
METRIC_TO_OPTIMISE = ['accuracy']
initial_assumption = {'path_to_model': './pretrain_model/resnet_base_pretrain.pt',
                      'model_type': 'ResNet18'}
scenario = 'from_checkpoint'
initial_assumption, learning_strategy = get_scenario_for_api(scenario, initial_assumption)

USER_CONFIG = {'problem': 'classification',
               'metric': METRIC_TO_OPTIMISE,
               'initial_assumption': initial_assumption,
               'pop_size': 1,
               'timeout': 0.1,
               'learning_strategy': learning_strategy,
               'learning_strategy_params': dict(epochs=15,
                                                learning_rate=0.0001,
                                                loss='crossentropy',
                                                custom_learning_params=dict(use_early_stopping={'patience': 30,
                                                                                                'maximise_task': False,
                                                                                                'delta': 0.01})
                                                ),
               'peft_strategy': 'quantization',
               'peft_strategy_params': dict(quant_type = 'qat', # 'static', 'dynamic', 'qat'
                                            allow_emb = False,
                                            allow_conv = True,
                                            qat_params={"epochs": 2,
                                                        "learning_rate": 0.001,
                                                        "loss": 'crossentropy'}
                                            )
}

if __name__ == "__main__":
    api_config = ApiConfigCheck().update_config_with_kwargs(DEFAULT_CLF_API_CONFIG, **USER_CONFIG)
    train_data, test_data = load_data(DATASET)
    fedcore_compressor = FedCore(**api_config)
    fedcore_model = fedcore_compressor.fit(train_data)
    model_comparison = fedcore_compressor.get_report(test_data)
    print(model_comparison['quality_comparasion'])