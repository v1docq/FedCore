import sys
sys.path.append('/Users/technocreep/Desktop/working-folder/fedot-core/FedCore')

from fedcore.api.config_factory import ConfigFactory

from fedcore.api.api_configs import APIConfigTemplate, AutoMLConfigTemplate, FedotConfigTemplate, LearningConfigTemplate, ModelArchitectureConfigTemplate, NeuralModelConfigTemplate, PruningTemplate
from fedcore.architecture.dataset.api_loader import ApiLoader
from fedcore.tools.example_utils import get_scenario_for_api
from fedcore.api.main import FedCore
# from fedcore.api.utils.checkers_collection import ApiConfigCheck
# from fedcore.data.dataloader import load_data
# from fedcore.repository.config_repository import DEFAULT_CLF_API_CONFIG

METRIC_TO_OPTIMISE = ['accuracy', 'latency']
LOSS = 'cross_entropy'
PROBLEM = 'classification'
PEFT_PROBLEM = 'pruning'
INITIAL_ASSUMPTION = {'path_to_model': '/Users/technocreep/Desktop/working-folder/fedot-core/FedCore/examples/api_example/pruning/pretrain_model/pretrain_model_checkpoint_at_15_epoch.pt',
                      'model_type': 'ResNet18'}
INITIAL_MODEL = 'ResNet18'
PRETRAIN_SCENARIO = 'from_checkpoint'
SCRATCH = 'from_scratch'


POP_SIZE = 2

def create_usage_scenario(scenario: str, model: str, path_to_pretrain: str = None):
    if path_to_pretrain is not None:
        initial_assumption = {'path_to_model': path_to_pretrain,
                              'model_type': model}
    else:
        initial_assumption = model
    return get_scenario_for_api(scenario, initial_assumption)

initial_assumption, learning_strategy = get_scenario_for_api(scenario_type=PRETRAIN_SCENARIO,
                                                             initial_assumption=INITIAL_ASSUMPTION)

model_config = ModelArchitectureConfigTemplate(input_dim=None,
                                               output_dim=None,
                                               depth=6)


pretrain_config = NeuralModelConfigTemplate(epochs=200,
                                            log_each=10,
                                            eval_each=15,
                                            save_each=50,
                                            criterion=LOSS,
                                            model_architecture=model_config,
                                            custom_learning_params=dict(use_early_stopping={'patience': 30,
                                                                                            'maximise_task': False,
                                                                                            'delta': 0.01}))

fedot_config = FedotConfigTemplate(problem=PROBLEM,
                                   metric=METRIC_TO_OPTIMISE,
                                   pop_size=POP_SIZE,
                                   timeout=5,
                                   initial_assumption=initial_assumption)

automl_config = AutoMLConfigTemplate(fedot_config=fedot_config)

finetune_config = NeuralModelConfigTemplate(epochs=3,
                                            log_each=3,
                                            eval_each=3,
                                            criterion=LOSS,
                                            )
peft_config = PruningTemplate(
    importance="magnitude",
    pruning_ratio=0.8,
    finetune_params=finetune_config
)
# peft_config = PruningTemplate(importance="GroupNormImportance",
#                               pruning_ratio=0.9,
#                               finetune_params=finetune_config,
#                               pruner_name='meta_pruner',
#                               importance_norm=2,
#                               finetune_params={'epochs': 1,
#                                                'learning_rate': 0.0001,
#                                                'loss': 'crossentropy'})


learning_config = LearningConfigTemplate(criterion=LOSS,
                                         learning_strategy=learning_strategy,
                                         learning_strategy_params=pretrain_config,
                                         peft_strategy=PEFT_PROBLEM,
                                         peft_strategy_params=peft_config)

DATASET = 'CIFAR10'
DATASET_PARAMS = {'train_bs': 64,
                  'val_bs': 100,
                  'train_shuffle': True,
                  'val_shuffle': False}
# initial_assumption = {'path_to_model': './pretrain_model/pretrain_model_checkpoint_at_15_epoch.pt',
#                       'model_type': 'ResNet18'}
# scenario = 'from_scratch'
# initial_assumption, learning_strategy = get_scenario_for_api(scenario, initial_assumption)

api_template = APIConfigTemplate(automl_config=automl_config,
                                 learning_config=learning_config)


if __name__ == "__main__":
    # api_config = ApiConfigCheck().update_config_with_kwargs(DEFAULT_CLF_API_CONFIG, **USER_CONFIG)
    al = ApiLoader('CIFAR10', {'split_ratio': [0.6, 0.4]})
    input_data = al._convert_to_fedcore(al._init_pretrain_dataset(al.source))
    
    APIConfig = ConfigFactory.from_template(api_template)
    api_config = APIConfig()
    fedcore_compressor = FedCore(api_config)
    fedcore_compressor.fit(input_data)
    # model_comparison = fedcore_compressor.get_report(fedcore_test_data)
    
    
    # # train_data, test_data = load_data(DATASET)
    # fedcore_compressor = FedCore(api_config)
    # fedcore_model = fedcore_compressor.fit(input_data=input_data)
    # model_comparison = fedcore_compressor.get_report(test_data)
    #onnx_model = fedcore_compressor.export()
