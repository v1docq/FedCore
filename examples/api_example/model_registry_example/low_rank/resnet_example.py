import sys
import os
import torch
import time
import gc
import logging

# Правильный путь с учетом вложенности
correct_path = "/home/user/projects/FedCore/FedCore"
sys.path.insert(0, correct_path)

from fedcore.api.config_factory import ConfigFactory
from fedcore.api.api_configs import (APIConfigTemplate, AutoMLConfigTemplate, FedotConfigTemplate,
                                     LearningConfigTemplate, ModelArchitectureConfigTemplate,
                                     NeuralModelConfigTemplate, LowRankTemplate)
from fedcore.architecture.dataset.api_loader import ApiLoader
from fedcore.data.dataloader import load_data
from fedcore.tools.example_utils import get_scenario_for_api
from fedcore.api.main import FedCore
from fedcore.tools.registry.model_registry import ModelRegistry

log_dir = 'examples/api_example/model_registry_example/logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'LR_resnet.log')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_file, mode='w')
console_handler = logging.StreamHandler()
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(log_format)
console_handler.setFormatter(log_format)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

METRIC_TO_OPTIMISE = ['accuracy', 'f1']
LOSS = 'cross_entropy'
PROBLEM = 'classification'
PEFT_PROBLEM = 'low_rank'
INITIAL_ASSUMPTION = {'path_to_model': 'examples/api_example/pruning/cv_task/pretrain_models/pretrain_model_checkpoint_at_15_epoch.pt',
                      'model_type': 'ResNet18'}
train_dataloader_params = {"batch_size": 64,
                           'shuffle': True,
                           'is_train': True,
                           'data_type': 'table',
                           'split_ratio': [0.8, 0.2]}
test_dataloader_params = {"batch_size": 100,
                          'shuffle': True,
                          'is_train': False,
                          'data_type': 'table'}


def create_usage_scenario(scenario: str, model: str, path_to_pretrain: str = None):
    if path_to_pretrain is not None:
        initial_assumption = {'path_to_model': path_to_pretrain,
                              'model_type': model}
    else:
        initial_assumption = model
    return get_scenario_for_api(scenario, initial_assumption)


def load_benchmark_dataset(dataset_name, train_dataloader_params, test_dataloader_params):
    fedcore_train_data = load_data(source=dataset_name, loader_params=train_dataloader_params)
    fedcore_test_data = load_data(source=dataset_name, loader_params=test_dataloader_params)
    return fedcore_train_data, fedcore_test_data

################################################################################
### CREATE SCENARIO FOR FEDCORE AGENT (TRAIN AND OPTIMISE MODEL FROM SCRATCH ###
### or optimise pretrained model with PEFT strategies                        ###
################################################################################

model_config = ModelArchitectureConfigTemplate(input_dim=None,
                                               output_dim=None,
                                               depth=6)

pretrain_config = NeuralModelConfigTemplate(epochs=5,
                                            log_each=10,
                                            eval_each=15,
                                            save_each=50,
                                            criterion='cross_entropy',
                                            model_architecture=model_config,
                                            custom_learning_params=dict(use_early_stopping={'patience': 30,
                                                                                            'maximise_task': False,
                                                                                            'delta': 0.01}))
peft_config = LowRankTemplate(
    strategy='quantile',
    rank_prune_each=1, 
    custom_criterions=None,
    non_adaptive_threshold=0.3,  
    epochs=5,
    log_each=1,
    eval_each=1,
    decomposer='svd', 
    rank=None,  
    distortion_factor=0.6, 
)

fedot_config = FedotConfigTemplate(problem='classification',
                                   metric=METRIC_TO_OPTIMISE,
                                   pop_size=1,
                                   timeout=1,
                                   initial_assumption=INITIAL_ASSUMPTION)

automl_config = AutoMLConfigTemplate(fedot_config=fedot_config)

learning_config = LearningConfigTemplate(criterion='cross_entropy',
                                         learning_strategy='from_checkpoint',
                                         learning_strategy_params=pretrain_config,
                                         peft_strategy='low_rank',
                                         peft_strategy_params=peft_config)

api_template = APIConfigTemplate(automl_config=automl_config,
                                 learning_config=learning_config)

if __name__ == "__main__":
    registry = ModelRegistry(auto_cleanup=True)
    registry.force_cleanup()
    
    initial_memory = registry.get_memory_stats()
    
    start_init = time.time()
    APIConfig = ConfigFactory.from_template(api_template)
    api_config = APIConfig()
    fedcore_compressor = FedCore(api_config)
    init_time = time.time() - start_init
    
    start_data = time.time()
    fedcore_train_data, fedcore_test_data = load_benchmark_dataset('CIFAR10', train_dataloader_params,
                                                                   test_dataloader_params)
    data_load_time = time.time() - start_data
    memory_after_data = registry.get_memory_stats()
    
    start_fit = time.time()
    fedcore_compressor.fit(fedcore_train_data)
    fit_time = time.time() - start_fit
    memory_after_training = registry.get_memory_stats()
    
    start_report = time.time()
    model_comparison = fedcore_compressor.get_report(fedcore_test_data)
    report_time = time.time() - start_report
    memory_after_report = registry.get_memory_stats()
    
    memory_before_cleanup = registry.get_memory_stats()
    logger.info(f"Memory before cleanup: {memory_before_cleanup.get('allocated_gb', 0):.4f} GB")
    
    fedcore_id = None
    if hasattr(fedcore_compressor, 'fedcore_model') and fedcore_compressor.fedcore_model is not None:
        pipeline = fedcore_compressor.fedcore_model
        if hasattr(pipeline, 'operator') and hasattr(pipeline.operator, 'root_node'):
            fitted_op = getattr(pipeline.operator.root_node, 'fitted_operation', None)
            if fitted_op is not None:
                fedcore_id = getattr(fitted_op, '_fedcore_id', None)
        
        if fedcore_id is None:
            logger.warning("Could not find _fedcore_id in fedcore_model")
        
        logger.info(f"Using fedcore_id: {fedcore_id}")
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