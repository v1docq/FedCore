import os
import sys
import time
import gc
import uuid

import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

correct_path = "/home/user/projects/FedCore/FedCore"
sys.path.insert(0, correct_path)

from fedcore.api.main import FedCore
from fedcore.api.config_factory import ConfigFactory
from fedcore.api.api_configs import (APIConfigTemplate, LearningConfigTemplate, 
                                     AutoMLConfigTemplate, DeviceConfigTemplate, 
                                     ComputeConfigTemplate, FedotConfigTemplate,
                                     LowRankTemplate)
from fedcore.api.llm_config import LLMConfigTemplate
from fedcore.data.dataloader import load_data
from fedcore.repository.constant_repository import Task, TaskTypesEnum
from fedcore.data.data import CompressionInputData
from fedcore.tools.registry.model_registry import ModelRegistry

log_dir = 'examples/api_example/model_registry_example/logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'LR_pretrain_transformer.log')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_file, mode='w')
console_handler = logging.StreamHandler(sys.stdout)
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(log_format)
console_handler.setFormatter(log_format)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


##########################################################################
### CONFIGURATION ###
##########################################################################

INITIAL_MODEL = "arnir0/Tiny-LLM"  

def preprocess_dataset(examples):
    """Extract text from different dataset formats"""
    if 'text' in examples:
        return {"text": examples["text"]}
    else:
        for key, value in examples.items():
            if isinstance(value[0], str):
                return {"text": examples[key]}
        return {"text": [str(x) for x in examples[list(examples.keys())[0]]]}

def tokenize_function(examples, tokenizer):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

class WikitextDataset(torch.utils.data.Dataset):
    """Dataset that returns tuples (input_ids, labels, attention_mask) for compatibility with LowRankModel and LLMTrainer"""
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]

        return (
            item['input_ids'], 
            item['labels'],            
            item['attention_mask']    
        )

def collate_fn(batch):
    """
    Custom collate function that handles batching for LLM data.
    Converts tuple format (input_ids, labels, attention_mask) to stacked tensors.
    
    LLMTrainer will convert this back to dict format in _dataloader_to_dataset.
    """
    input_ids_list = []
    labels_list = []
    attention_mask_list = []
    
    for item in batch:
        input_ids = torch.tensor(item[0]) if isinstance(item[0], list) else item[0]
        labels = torch.tensor(item[1]) if isinstance(item[1], list) else item[1]
        attention_mask = torch.tensor(item[2]) if isinstance(item[2], list) else item[2]
        
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)
    
    input_ids_batch = torch.stack(input_ids_list)
    labels_batch = torch.stack(labels_list)
    attention_mask_batch = torch.stack(attention_mask_list)
 
    return input_ids_batch, labels_batch, attention_mask_batch

def create_pretrain_config(model, tokenizer):
    """Create pretraining configuration."""
    return LLMConfigTemplate(
        epochs=3,  
        optimizer='adamw',
        scheduler='one_cycle',
        scheduler_step_each=1,
        criterion='cross_entropy',
        is_llm=True,
        model=model,
        tokenizer=tokenizer,
        custom_learning_params={
            'batch_size': 2,  
            'learning_rate': 5e-5,
            'warmup_steps': 10,
            'logging_steps': 5,
            'save_steps': 50
        }
    )

def create_peft_config(fedcore_id=None):
    """Create PEFT configuration."""
    return LowRankTemplate(
        strategy='quantile',
        rank_prune_each=1, 
        custom_criterions=None,
        non_adaptive_threshold=0.3,  
        epochs=3,  
        log_each=1,
        eval_each=1,
        decomposer='svd',  
        rank=None,  
        distortion_factor=0.6,
        fedcore_id=fedcore_id,
    )

def create_api_config(model, tokenizer, fedcore_id=None):
    """Create API configuration with all dependencies"""
    pretrain_config = create_pretrain_config(model, tokenizer)
    peft_config = create_peft_config(fedcore_id)
    
    fedot_config = FedotConfigTemplate(
        problem='classification',
        metric=['accuracy', 'f1'],
        pop_size=1,
        timeout=2,
        initial_assumption=model
    )
    
    automl_config = AutoMLConfigTemplate(fedot_config=fedot_config)
    
    learning_config = LearningConfigTemplate(
        criterion='cross_entropy',
        learning_strategy='from_scratch',
        learning_strategy_params=pretrain_config,
        peft_strategy='low_rank', 
        peft_strategy_params=peft_config  
    )
    
    device_config = DeviceConfigTemplate(device='cuda' if torch.cuda.is_available() else 'cpu')
    compute_config = ComputeConfigTemplate(
        output_folder='./llm_low_rank_pretrain_experiment',
        automl_folder='./llm_low_rank_pretrain_automl'
    )
    
    api_template = APIConfigTemplate(
        device_config=device_config,
        automl_config=automl_config,
        learning_config=learning_config,
        compute_config=compute_config
    )
    
    return api_template

def load_benchmark_dataset(dataset_name, train_dataloader_params, test_dataloader_params):
    """Load benchmark dataset."""
    fedcore_train_data = load_data(source=dataset_name, loader_params=train_dataloader_params)
    fedcore_test_data = load_data(source=dataset_name, loader_params=test_dataloader_params)
    return fedcore_train_data, fedcore_test_data

if __name__ == "__main__":
    # auto_cleanup=False for LLM experiments to avoid clearing optimizer/scheduler during training
    registry = ModelRegistry(auto_cleanup=False)
    logger.info(f"ModelRegistry initialized with auto_cleanup={registry.auto_cleanup}")
    
    start_init = time.time()
    logger.info("GPU INITIAL STATE:")
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1024**3
        reserved_gb = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"  allocated_gb: {allocated_gb:.4f} GB")
        logger.info(f"  reserved_gb: {reserved_gb:.4f} GB")
    else:
        logger.info("  CUDA not available")
    
    logger.info(f"Loading model from HuggingFace: {INITIAL_MODEL}")
    
    model = AutoModelForCausalLM.from_pretrained(INITIAL_MODEL)
    logger.info(f"Model loaded: {type(model).__name__}")
    initial_memory = registry.get_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(INITIAL_MODEL)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    logger.info(f"Tokenizer loaded. Vocabulary size: {tokenizer.vocab_size}")
    
    fedcore_id = f"fedcore_{uuid.uuid4().hex[:8]}"
    logger.info(f"Generated fedcore_id: {fedcore_id}")
    api_template = create_api_config(model, tokenizer, fedcore_id)
    
    APIConfig = ConfigFactory.from_template(api_template)
    api_config = APIConfig()
    fedcore_compressor = FedCore(api_config)
    init_time = time.time() - start_init
    logger.info(f"Initialization time: {init_time:.4f} sec")
    
    if hasattr(fedcore_compressor, 'fedcore_model'):
        model_class = fedcore_compressor.fedcore_model.__class__.__name__
        logger.info(f"Trainer class: {model_class}")
        
        if hasattr(fedcore_compressor.fedcore_model, 'operation_impl'):
            trainer_type = type(fedcore_compressor.fedcore_model.operation_impl).__name__
            logger.info(f"Trainer type: {trainer_type}")
    
    logger.info("GPU AFTER INITIALIZATION:")
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1024**3
        reserved_gb = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"  allocated_gb: {allocated_gb:.4f} GB")
        logger.info(f"  reserved_gb: {reserved_gb:.4f} GB")
    
    start_data = time.time()
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    
    dataset = dataset.map(preprocess_dataset, batched=True)
    dataset = dataset.filter(lambda example: len(example["text"].strip()) > 0)
    logger.info(f"Dataset filtered: {len(dataset)} samples")
    
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer), 
        batched=True
    )
    
    wikitext_dataset = WikitextDataset(tokenized_dataset)
    train_size = int(0.8 * len(wikitext_dataset))
    val_size = int(0.1 * len(wikitext_dataset))
    test_size = len(wikitext_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        wikitext_dataset, [train_size, val_size, test_size]
    )
    logger.info(f"Datasets split: train={train_size}, val={val_size}, test={test_size}")
    
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    example_batch = next(iter(val_dataloader))
    example_input = example_batch[0]
    logger.info(f"Example batch shape: {example_input.shape}")
    
    compression_data = CompressionInputData(
        features=example_input,  
        target=model, 
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        task=Task(TaskTypesEnum.classification),  
        input_dim=tokenizer.vocab_size,
    )
    data_load_time = time.time() - start_data
    
    logger.info("GPU AFTER DATA PREPARATION:")
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1024**3
        reserved_gb = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"  allocated_gb: {allocated_gb:.4f} GB")
        logger.info(f"  reserved_gb: {reserved_gb:.4f} GB")

    start_fit = time.time()
    fedcore_compressor.fit_no_evo(compression_data)
    fit_time = time.time() - start_fit
    memory_after_training = registry.get_memory_stats()
    
    start_report = time.time()
    model_comparison = fedcore_compressor.get_report(compression_data)
    report_time = time.time() - start_report
    memory_after_report = registry.get_memory_stats()
    
    memory_before_cleanup = registry.get_memory_stats()
    
    fedcore_id = None
    if hasattr(fedcore_compressor, 'fedcore_model') and fedcore_compressor.fedcore_model is not None:
        fedcore_id = fedcore_compressor.fedcore_model._fedcore_id
        registry.cleanup_fedcore_instance(fedcore_id, fedcore_compressor.fedcore_model)
    else:
        registry.force_cleanup()
    
    final_memory = registry.get_memory_stats()
    
    logger.info("GPU AFTER CLEANUP:")
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1024**3
        reserved_gb = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"  allocated_gb: {allocated_gb:.4f} GB")
        logger.info(f"  reserved_gb: {reserved_gb:.4f} GB")
    
    logger.info("FINAL STATISTICS")
    logger.info(f"Initialization time: {init_time:.2f} sec")
    logger.info(f"Data loading time:   {data_load_time:.2f} sec")
    logger.info(f"Training time:       {fit_time:.2f} sec")
    logger.info(f"Report time:         {report_time:.2f} sec")
    logger.info(f"Total time:          {init_time + data_load_time + fit_time + report_time:.2f} sec")
    
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
