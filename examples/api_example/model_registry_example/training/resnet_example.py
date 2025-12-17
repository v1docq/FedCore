import sys
import os
import torch
import time
import gc
import uuid
import logging

# Правильный путь с учетом вложенности
correct_path = "/home/user/projects/FedCore/FedCore"
sys.path.insert(0, correct_path)

from fedcore.api.config_factory import ConfigFactory
from fedcore.api.api_configs import (APIConfigTemplate, AutoMLConfigTemplate, FedotConfigTemplate,
                                     LearningConfigTemplate, ModelArchitectureConfigTemplate,
                                     NeuralModelConfigTemplate)
from fedcore.data.dataloader import load_data
from fedcore.tools.example_utils import get_scenario_for_api
from fedcore.api.main import FedCore
from fedcore.tools.registry.model_registry import ModelRegistry
from fedot.core.repository.metrics_repository import ClassificationMetricsEnum


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

for logger_name in [
    'fedcore.interfaces.fedcore_optimizer',
    'fedcore.repository.fedcore_impl.metrics',
    'FedcoreEvoOptimizer',
    'MetricsObjective',
    'fedcore.metrics.metric_impl',
]:
    logging.getLogger(logger_name).setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

log_dir = 'examples/api_example/model_registry_example/logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'training_resnet.log')

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_file, mode='w')
console_handler = logging.StreamHandler(sys.stdout)
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(log_format)
console_handler.setFormatter(log_format)

# logger.addHandler(file_handler)
# logger.addHandler(console_handler)

# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(sys.stdout)
#     ]
# )


##########################################################################
### DEFINE ML PROBLEM (classification) and appropriate loss function  ###
### Training from scratch with BaseNeuralModel (peft_strategy='training')
##########################################################################
METRIC_TO_OPTIMISE = [
    ClassificationMetricsEnum.accuracy,
    ClassificationMetricsEnum.f1,
]
LOSS = 'cross_entropy'
PROBLEM = 'classification'
INITIAL_ASSUMPTION = 'ResNet18'

train_dataloader_params = {"batch_size": 64,
                           'shuffle': True,
                           'is_train': True,
                           'data_type': 'table',
                           'split_ratio': [0.8, 0.2]}
test_dataloader_params = {"batch_size": 100,
                          'shuffle': True,
                          'is_train': False,
                          'data_type': 'table'}


def create_usage_scenario(scenario: str, model: str):
    initial_assumption = model
    return get_scenario_for_api(scenario, initial_assumption)


def load_benchmark_dataset(dataset_name, train_dataloader_params, test_dataloader_params):
    logger.debug(f'Loading dataset {dataset_name} with train params {train_dataloader_params} and test params {test_dataloader_params}')
    fedcore_train_data = load_data(source=dataset_name, loader_params=train_dataloader_params)
    fedcore_test_data = load_data(source=dataset_name, loader_params=test_dataloader_params)
    train_info = f'train_dataloader={fedcore_train_data.train_dataloader is not None}, val_dataloader={fedcore_train_data.val_dataloader is not None}'
    test_info = f'test_dataloader={fedcore_test_data.test_dataloader is not None}, val_dataloader={fedcore_test_data.val_dataloader is not None}'
    logger.debug(f'Dataset loaded: train_data ({train_info}), test_data ({test_info})')
    return fedcore_train_data, fedcore_test_data

################################################################################
### CREATE SCENARIO FOR FEDCORE AGENT - TRAIN MODEL FROM SCRATCH            ###
### WITHOUT PEFT strategies, only BaseNeuralModel training                  ###
################################################################################

def create_api_template(fedcore_id=None):
    """Create API template with fedcore_id for model registration."""
    model_config = ModelArchitectureConfigTemplate(input_dim=None,
                                                   output_dim=None,
                                                   depth=6)

    train_config = NeuralModelConfigTemplate(epochs=2,
                                             log_each= 1,
                                             eval_each=1,
                                             save_each=50,
                                             criterion='cross_entropy',
                                             model_architecture=model_config,
                                             custom_learning_params=dict(use_early_stopping={'patience': 30,
                                                                                             'maximise_task': False,
                                                                                             'delta': 0.01}))

    fedot_config = FedotConfigTemplate(problem='classification',
                                       metric=METRIC_TO_OPTIMISE,
                                       pop_size=1,
                                       timeout=.1,
                                       initial_assumption=INITIAL_ASSUMPTION)

    automl_config = AutoMLConfigTemplate(fedot_config=fedot_config)

    learning_config = LearningConfigTemplate(criterion='cross_entropy',
                                             learning_strategy='checkpoint',
                                             learning_strategy_params=train_config,
                                             peft_strategy='training',
                                             peft_strategy_params=train_config,
                                             fedcore_id=fedcore_id)

    return APIConfigTemplate(automl_config=automl_config,
                            learning_config=learning_config)

if __name__ == "__main__":
    registry = ModelRegistry(auto_cleanup=True)
    registry.force_cleanup()

    initial_memory = registry.get_memory_stats()

    fedcore_id = f"fedcore_{uuid.uuid4().hex[:8]}"
    logger.info(f"Generated fedcore_id: {fedcore_id}")

    start_init = time.time()
    api_template = create_api_template(fedcore_id)
    APIConfig = ConfigFactory.from_template(api_template)
    api_config = APIConfig()
    logger.debug(f'API config initialised: {api_config}')
    fedcore_compressor = FedCore(api_config)
    logger.debug('FedCore instance created')
    init_time = time.time() - start_init

    start_data = time.time()
    fedcore_train_data, fedcore_test_data = load_benchmark_dataset('CIFAR10', train_dataloader_params,
                                                                   test_dataloader_params)
    data_load_time = time.time() - start_data
    memory_after_data = registry.get_memory_stats()

    start_fit = time.time()
    try:
        logger.debug('Starting FedCore.fit')
        fedcore_compressor.fit(fedcore_train_data)
        logger.debug('FedCore.fit finished successfully')
    except Exception as fit_error:
        solver = getattr(getattr(fedcore_compressor, 'manager', None), 'solver', None)
        current_pipeline = getattr(solver, 'current_pipeline', None) if solver else None
        logger.exception('FedCore.fit failed: %s | solver=%s | pipeline=%s | history=%s',
                         fit_error,
                         type(solver).__name__ if solver else None,
                         getattr(current_pipeline, 'print_structure', lambda: current_pipeline)(),
                         getattr(getattr(solver, 'history', None), 'generations', None)
                         )
        raise
    fit_time = time.time() - start_fit
    memory_after_training = registry.get_memory_stats()

    extracted_fedcore_id = None
    if hasattr(fedcore_compressor, 'fedcore_model') and fedcore_compressor.fedcore_model is not None:
        extracted_fedcore_id = getattr(fedcore_compressor.fedcore_model, '_fedcore_id', None)

    if hasattr(fedcore_compressor, 'fedcore_model') and fedcore_compressor.fedcore_model is not None:
        trained_model = getattr(fedcore_compressor.fedcore_model, 'model', None)
        if trained_model is not None:
            try:
                model_id = registry.register_model(
                    fedcore_id=fedcore_id,
                    model=trained_model,
                    metrics={"operation": "training", "stage": "after_training"}
                )
                logger.info(f"Trained model registered with ID: {model_id}")
                logger.info(f"Using fedcore_id: {fedcore_id}")
            except Exception as e:
                logger.warning(f"Failed to save trained model to registry: {e}")

    fedcore_id = fedcore_id

    start_report = time.time()
    logger.debug('Generating report')
    model_comparison = fedcore_compressor.get_report(fedcore_test_data)
    report_time = time.time() - start_report

    memory_after_report = registry.get_memory_stats()

    logger.info("REGISTERED MODELS INFO:")
    logger.info(f"  FedCore ID: {fedcore_id}")
    if fedcore_id:
        model_ids = registry.list_models(fedcore_id)
        logger.info(f"  Number of registered models: {len(model_ids)}")
        if not model_ids:
            logger.info("  No models registered")
        for idx, model_id in enumerate(model_ids, 1):
            logger.info(f"  {idx}. Model ID: {model_id}")
            latest_record = registry.get_latest_record(fedcore_id, model_id)
            if latest_record:
                logger.info(f"     - Checkpoint path: {latest_record.get('checkpoint_path', 'N/A')}")
                logger.info(f"     - Stage: {latest_record.get('metrics', {}).get('stage', 'N/A')}")
            else:
                logger.info(f"     - No record found for this model")
    else:
        logger.info("  FedCore ID is None - cannot retrieve model information")

    memory_before_cleanup = registry.get_memory_stats()

    if hasattr(fedcore_compressor, 'fedcore_model') and fedcore_compressor.fedcore_model is not None:
        logger.info(f"Using fedcore_id: {fedcore_id}")
        logger.debug('Cleaning up fedcore instance')
        registry.cleanup_fedcore_instance(fedcore_id if fedcore_id else "unknown", fedcore_compressor.fedcore_model)
    else:
        registry.force_cleanup()

    final_memory = registry.get_memory_stats()
    logger.info(f"Memory after cleanup: {final_memory.get('allocated_gb', 0):.4f} GB")

    logger.info("FINAL STATISTICS")
    logger.info(f"Training time:     {fit_time:.2f} sec")
    logger.info(f"Report time:       {report_time:.2f} sec")
    logger.info(f"Total time:        {init_time + data_load_time + fit_time + report_time:.2f} sec")

    logger.info("MEMORY STATISTICS:")
    logger.info(f"Initial GPU memory:     {initial_memory.get('allocated_gb', 0):.4f} GB")
    logger.info(f"After training:          {memory_after_training.get('allocated_gb', 0):.4f} GB")
    logger.info(f"After report:            {memory_after_report.get('allocated_gb', 0):.4f} GB")
    logger.info(f"Final GPU memory:        {final_memory.get('allocated_gb', 0):.4f} GB")

    peak_memory = max(
        initial_memory.get('allocated_gb', 0),
        memory_after_training.get('allocated_gb', 0),
        memory_after_report.get('allocated_gb', 0)
    )
    memory_freed = peak_memory - final_memory.get('allocated_gb', 0)
    logger.info(f"Peak memory:             {peak_memory:.4f} GB")
    logger.info(f"Memory freed:             {memory_freed:.4f} GB")

    if peak_memory > 0:
        cleanup_percentage = (memory_freed / peak_memory) * 100
        logger.info(f"Cleanup efficiency:      {cleanup_percentage:.1f}%")
    
