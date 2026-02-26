import sys
import os
import torch
import torch.nn.utils.prune as prune
import time
import gc
import logging
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
correct_path = "/home/user/projects/FedCore/FedCore"
sys.path.insert(0, correct_path)
from fedcore.data.data import CompressionInputData
from fedcore.repository.constant_repository import FedotTaskEnum
from fedcore.models.network_impl.llm_trainer import LLMTrainer

log_dir = 'validation_experiment/logs'
results_dir = 'validation_experiment/results'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'prune_050_exp_logs.log')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_file, mode='w')
console_handler = logging.StreamHandler(sys.stdout)
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(log_format)
console_handler.setFormatter(log_format)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info(f"LOGGING TEST - logs will be saved to: {log_file}")

##########################################################################
### DEFINE ML PROBLEM
##########################################################################
METRIC_TO_OPTIMISE = ['Perplexity']  
LOSS = 'cross_entropy' 
PROBLEM = 'classification'
DATASET_NAME = "imdb"

INITIAL_MODEL = 'Qwen/Qwen2.5-0.5B'

model = AutoModelForCausalLM.from_pretrained(INITIAL_MODEL, num_labels=2, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(INITIAL_MODEL)
tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
model.config.pad_token_id = tokenizer.pad_token_id

INITIAL_ASSUMPTION = model

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    """Tokenize text for causal language modeling"""
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256,  
    )
    
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

class CausalLMDataset(torch.utils.data.Dataset):
    """Dataset for causal language modeling"""
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        
        return (
            torch.tensor(item['input_ids'], dtype=torch.long),
            torch.tensor(item['labels'], dtype=torch.long),  # Same as input_ids
            torch.tensor(item['attention_mask'], dtype=torch.long)
        )


def collate_fn(batch):
    """
    Custom collate function for causal LM data.
    Returns input_ids, labels, attention_mask as stacked tensors.
    """
    input_ids_list = []
    labels_list = []
    attention_mask_list = []
    
    for item in batch:
        input_ids = item[0] if isinstance(item[0], torch.Tensor) else torch.tensor(item[0])
        labels = item[1] if isinstance(item[1], torch.Tensor) else torch.tensor(item[1])
        attention_mask = item[2] if isinstance(item[2], torch.Tensor) else torch.tensor(item[2])
        
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)
    
    input_ids_batch = torch.stack(input_ids_list)
    labels_batch = torch.stack(labels_list)
    attention_mask_batch = torch.stack(attention_mask_list)
    
    return {
        "input_ids": input_ids_batch,
        "labels": labels_batch,
        "attention_mask": attention_mask_batch
    }

def load_benchmark_dataset(
    dataset_name='imdb', 
    batch_size=4, 
    num_samples=None,
    max_length=512
):
    """
    Load and prepare dataset for causal language modeling
    """
    dataset = load_dataset(dataset_name)
    
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"] 
    )
    
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]
    
    if num_samples:
        train_dataset = train_dataset.select(range(min(num_samples, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(num_samples // 10, len(eval_dataset))))
    
    train_dataset = train_dataset.shuffle(seed=42)
    eval_dataset = eval_dataset.shuffle(seed=42)
    
    train_dataloader = DataLoader(
        CausalLMDataset(train_dataset),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        CausalLMDataset(eval_dataset),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    test_dataloader = DataLoader(
        CausalLMDataset(tokenized_datasets["test"]),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    

    fedcore_train_data = CompressionInputData(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        task="text_generation",
        num_classes=tokenizer.vocab_size,  
        input_dim=tokenizer.vocab_size,
    )
    
    fedcore_test_data = CompressionInputData(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        task="text_generation",
        num_classes=tokenizer.vocab_size,
        input_dim=tokenizer.vocab_size,
    )
    
    return fedcore_train_data, fedcore_test_data


def _to_scalar(x):
    """Привести step/loss к типу для записи в CSV (int/float)."""
    if hasattr(x, 'item'):
        return x.item()
    return float(x) if isinstance(x, (np.floating, float)) else int(x)

if __name__ == "__main__":
    fedcore_train_data, fedcore_test_data = load_benchmark_dataset('imdb', batch_size=4, num_samples=None)
    
    logger.info("Applying pruning to model before training...")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.5)
            prune.remove(module, 'weight')
    
    trainer = LLMTrainer(
            model=model,
            params={
                'tokenizer': tokenizer,
                'num_train_epochs': 1,
                'per_device_train_batch_size': 4,
                'per_device_eval_batch_size': 4,
                'output_dir': './output',
                'logging_steps': 500,
                'eval_steps': 500,
                'save_steps': 500,
                'num_reference_batches': 50,
            }
        )
    
    trained_model = trainer.fit(fedcore_train_data)
    
    ref_loss_history = trainer.history.get('ref_loss', [])
    

    train_loss_by_step = {}
    val_loss_by_step = {}

    internal_trainer = trainer._trainer if hasattr(trainer, '_trainer') else trainer.trainer_objects.get('trainer')

    if internal_trainer and hasattr(internal_trainer, 'state') and hasattr(internal_trainer.state, 'log_history'):
        log_history = internal_trainer.state.log_history
            
        if log_history:
            logger.info(f"First log entry keys: {list(log_history[0].keys())}")
            logger.info(f"Sample entries with step 500:")
            for entry in log_history:
                if entry.get('step') == 500:
                    logger.info(f"  Step 500 entry: {entry}")
            
        for log_entry in log_history:
            step = log_entry.get('step', None)
                
            if step is not None:
                step = int(step) 
                    
                if 'loss' in log_entry and 'eval_loss' not in log_entry:
                    train_loss_by_step[step] = _to_scalar(log_entry['loss'])
                elif 'loss' in log_entry and 'eval_loss' in log_entry:
                    train_loss_by_step[step] = _to_scalar(log_entry['loss'])
                    val_loss_by_step[step] = _to_scalar(log_entry['eval_loss'])
                    
                if 'eval_loss' in log_entry:
                    val_loss_by_step[step] = _to_scalar(log_entry['eval_loss'])

    all_steps = set()
    all_steps.update([int(step) for step, _ in ref_loss_history]) 
    all_steps.update(train_loss_by_step.keys())
    all_steps.update(val_loss_by_step.keys())
    all_steps = sorted(all_steps)

    results_data = []
    for step in all_steps:
        row = {'step': step}
            
        ref_loss_value = next((loss for s, loss in ref_loss_history if int(s) == step), None)
        row['ref_loss'] = _to_scalar(ref_loss_value) if ref_loss_value is not None else None
            
        row['train_loss'] = train_loss_by_step.get(step)
        row['val_loss'] = val_loss_by_step.get(step)
            
        results_data.append(row)

    df = pd.DataFrame(results_data)
    csv_path = os.path.join(results_dir, 'prune_050_res.csv')
    csv_path_abs = os.path.abspath(csv_path)
    df.to_csv(csv_path_abs, index=False)

    sample_rows = df[df[['ref_loss', 'train_loss', 'val_loss']].notna().all(axis=1)].head(3)
    logger.info(f"\n{sample_rows.to_string()}")

