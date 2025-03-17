from fedcore.tools.example_utils import get_scenario_for_api
from fedcore.api.main import FedCore
from fedcore.api.utils.checkers_collection import ApiConfigCheck
from fedcore.data.dataloader import load_data
from fedcore.api.utils.evaluation import evaluate_original_model, evaluate_optimised_model
from fedcore.repository.config_repository import DEFAULT_CLF_API_CONFIG

DATASET = 'CIFAR10'
DATASET_PARAMS = {'train_bs': 64,
                  'val_bs': 100,
                  'train_shuffle': True,
                  'val_shuffle': False}
METRIC_TO_OPTIMISE = ['accuracy', 'latency']
initial_assumption = {'path_to_model': './pretrain_model/pretrain_model_checkpoint_at_15_epoch.pt',
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
                                                # custom_loss = ['norm_loss', 'weight_loss'] needs to rework BaseNN class
                                                # custom_loss_params = {'norm_loss':{...},
                                                # 'weight_loss':{...}}
                                                custom_learning_params=dict(use_early_stopping={'patience': 30,
                                                                                                'maximise_task': False,
                                                                                                'delta': 0.01})
                                                ),
               'peft_strategy': 'pruning',
               'peft_strategy_params': dict(pruning_iterations=1,
                                            importance="GroupNormImportance", # "BNScaleImportance", 'MagnitudeImportance', "GroupNormImportance",
                                            pruner_name='meta_pruner',
                                            importance_norm=2,
                                            pruning_ratio=0.9,
                                            finetune_params={'epochs': 1,
                                                             "learning_rate": 0.0001,
                                                             'loss': 'crossentropy'}
                                            )
               }


def eval_and_compare_metrics(fedcore_compressor, pruning_prediction, original_prediction, test_data):
    quality_metrics_after_pruning = fedcore_compressor.evaluate_metric(predicton=pruning_prediction,
                                                                       target=test_data.target,
                                                                       problem='classification')
    computational_metrics_after_pruning = fedcore_compressor.evaluate_metric(
        predicton=pruning_prediction,
        target=test_data.calib_dataloader,
        problem="computational")
    # get prediction and quality metrics from pruned model

    quality_metrics_before_pruning = fedcore_compressor.evaluate_metric(predicton=original_prediction,
                                                                        target=test_data.target,
                                                                        problem='classification')
    computational_metrics_before_pruning = fedcore_compressor.evaluate_metric(
        predicton=original_prediction,
        target=test_data.calib_dataloader,
        problem="computational")
    return quality_metrics_after_pruning


if __name__ == "__main__":
    api_config = ApiConfigCheck().update_config_with_kwargs(DEFAULT_CLF_API_CONFIG, **USER_CONFIG)
    train_data, test_data = load_data(DATASET)
    fedcore_compressor = FedCore(**api_config)
    fedcore_model = fedcore_compressor.fit(train_data)
    # get prediction and quality metrics from pruned and original model
    pruning_prediction = fedcore_compressor.predict(test_data, output_mode="compress")
    original_prediction = fedcore_compressor.predict(test_data, output_mode="original")
    metric_comparison = eval_and_compare_metrics(fedcore_compressor, pruning_prediction, original_prediction, test_data)
    onnx_model = fedcore_compressor.export()
