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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

# Явно устанавливаем уровень для критичных логгеров
logging.getLogger('BaseCompressionModel').setLevel(logging.INFO)
logging.getLogger('BaseCompressionModel.__init__').setLevel(logging.INFO)
logging.getLogger('BaseCompressionModel._init_model').setLevel(logging.INFO)
logging.getLogger('BaseCompressionModel.model_before_setter').setLevel(logging.INFO)
logging.getLogger('BaseCompressionModel.model_after_setter').setLevel(logging.INFO)
logging.getLogger('LowRankModel._init_model').setLevel(logging.INFO)
logging.getLogger('LowRankModel').setLevel(logging.INFO)
logging.getLogger('ModelRegistry').setLevel(logging.INFO)

##########################################################################
### DEFINE ML PROBLEM (classification, object_detection, regression,   ###
### ts_forecasting), PEFT problem (pruning, quantisation, distillation,###
### low_rank) and appropriate loss function both for model and compute ###
##########################################################################
METRIC_TO_OPTIMISE = ['accuracy', 'latency']
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
                                   metric=['accuracy', 'latency'],
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
    logging.info(f"ModelRegistry initialized with auto_cleanup={registry.auto_cleanup}")
    registry.force_cleanup()
    
    logging.info("GPU INITIAL STATE:")
    initial_memory = registry.get_memory_stats()
    for key, value in initial_memory.items():
        if isinstance(value, float):
            logging.info(f"  {key}: {value:.4f} GB")
        else:
            logging.info(f"  {key}: {value}")
    
    start_init = time.time()
    APIConfig = ConfigFactory.from_template(api_template)
    api_config = APIConfig()
    fedcore_compressor = FedCore(api_config)
    init_time = time.time() - start_init
    logging.info(f"FedCore initialization time: {init_time:.4f} sec")
    
    start_data = time.time()
    fedcore_train_data, fedcore_test_data = load_benchmark_dataset('CIFAR10', train_dataloader_params,
                                                                   test_dataloader_params)
    data_load_time = time.time() - start_data
    logging.info(f"Data load time: {data_load_time:.4f} sec")

    logging.info("GPU AFTER DATA LOADING:")
    memory_after_data = registry.get_memory_stats()
    for key, value in memory_after_data.items():
        if isinstance(value, float):
            logging.info(f"  {key}: {value:.4f} GB")
        else:
            logging.info(f"  {key}: {value}")
    
    start_fit = time.time()
    fedcore_compressor.fit_no_evo(fedcore_train_data)
    fit_time = time.time() - start_fit
    logging.info(f"fit_no_evo time lead: {fit_time:.4f} sec")
    
    logging.info("GPU AFTER TRAINING:")
    memory_after_fit = registry.get_memory_stats()
    for key, value in memory_after_fit.items():
        if isinstance(value, float):
            logging.info(f"  {key}: {value:.4f} GB")
        else:
            logging.info(f"  {key}: {value}")
    
    start_report = time.time()
    model_comparison = fedcore_compressor.get_report(fedcore_test_data)
    report_time = time.time() - start_report
    logging.info(f"get_report lead time: {report_time:.4f} sec")
    
    logging.info("GPU AFTER get_report (модель загружена для оценки):")
    memory_after_report = registry.get_memory_stats()
    for key, value in memory_after_report.items():
        if isinstance(value, float):
            logging.info(f"  {key}: {value:.4f} GB")
        else:
            logging.info(f"  {key}: {value}")
    
    # Get fedcore_id before deleting the object
    fedcore_id = getattr(fedcore_compressor, '_fedcore_id', None)
    
    logging.info("Deleting FedCore object and performing final cleanup")
    del fedcore_compressor
    gc.collect()  
    registry.force_cleanup()  
    
    logging.info("GPU AFTER CLEANUP:")
    memory_after_cleanup = registry.get_memory_stats()
    for key, value in memory_after_cleanup.items():
        if isinstance(value, float):
            logging.info(f"  {key}: {value:.4f} GB")
        else:
            logging.info(f"  {key}: {value}")
    
    logging.info("REGISTERED MODELS INFO:")
    try:
        if fedcore_id:
            model_ids = registry.list_models(fedcore_id)
            logging.info(f"  Number of registered models: {len(model_ids)}")
            for idx, model_id in enumerate(model_ids, 1):
                logging.info(f"  {idx}. Model ID: {model_id}")
                latest_record = registry.get_latest_record(fedcore_id, model_id)
                if latest_record:
                    logging.info(f"     - Checkpoint path: {latest_record.get('checkpoint_path', 'N/A')}")
                    logging.info(f"     - Stage: {latest_record.get('metrics', {}).get('stage', 'N/A')}")
    except Exception as e:
        logging.warning(f"Unable to retrieve information about models: {e}")
    
    logger = logging.getLogger('EXPERIMENT_SUMMARY')
    logger.info("="*80)
    logger.info("FINAL STATISTICS")
    logger.info("="*80)
    
    logger.info("TIMING METRICS:")
    logger.info(f"  - Initialization:         {init_time:.4f} sec")
    logger.info(f"  - Data loading:           {data_load_time:.4f} sec")
    logger.info(f"  - Training/optimization:  {fit_time:.4f} sec")
    logger.info(f"  - Report generation:      {report_time:.4f} sec")
    logger.info(f"  - TOTAL TIME:             {init_time + data_load_time + fit_time + report_time:.4f} sec")
    
    if torch.cuda.is_available() and initial_memory:
        logger.info("GPU MEMORY CHANGES:")
        logger.info(f"  - Initial:                {initial_memory.get('allocated_gb', 0):.4f} GB")
        logger.info(f"  - After data loading:     {memory_after_data.get('allocated_gb', 0):.4f} GB")
        logger.info(f"  - After training:         {memory_after_fit.get('allocated_gb', 0):.4f} GB")
        logger.info(f"  - After get_report:       {memory_after_report.get('allocated_gb', 0):.4f} GB")
        logger.info(f"  - After final cleanup:    {memory_after_cleanup.get('allocated_gb', 0):.4f} GB")
        
        mem_initial = initial_memory.get('allocated_gb', 0)
        mem_after_data = memory_after_data.get('allocated_gb', 0)
        mem_after_train = memory_after_fit.get('allocated_gb', 0)
        mem_after_report = memory_after_report.get('allocated_gb', 0)
        mem_after_cleanup = memory_after_cleanup.get('allocated_gb', 0)
        
        logger.info("MEMORY CLEANUP ANALYSIS:")
        
        final_cleanup = mem_after_report - mem_after_cleanup
        
        total_used = mem_after_report 
        total_freed = mem_after_report - mem_after_cleanup
        
        logger.info(f"  - Peak memory usage:        {mem_after_report:.4f} GB (after get_report)")
        logger.info(f"  - Memory after training:    {mem_after_train:.4f} GB (after ModelRegistry cleanup)")
        logger.info(f"  - Final cleanup freed:      {final_cleanup:.4f} GB")
        logger.info(f"  - Remaining allocated:      {mem_after_cleanup:.4f} GB")
        
        if mem_after_train < mem_after_report:
            logger.info(f"  - get_report loaded models: {mem_after_report - mem_after_train:.4f} GB")
 
    else:
        logger.warning("CUDA unavailable - GPU memory statistics not available")