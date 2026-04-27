import sys
import os
import torch
import time
import gc
import logging
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
# Правильный путь с учетом вложенности
correct_path = "/home/user/projects/FedCore/FedCore"
sys.path.insert(0, correct_path)

from fedcore.data.data import CompressionInputData
from fedcore.repository.constant_repository import FedotTaskEnum
from fedcore.models.network_impl.llm_trainer import LLMTrainer

_script_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(_script_dir, 'logs')
results_dir = os.path.join(_script_dir, 'results')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'validation_exp_logs.log')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
console_handler = logging.StreamHandler(sys.stdout)
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(log_format)
console_handler.setFormatter(log_format)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info(f"LOGGING TEST - logs will be saved to: {log_file}")
file_handler.flush()

##########################################################################
### DEFINE ML PROBLEM
##########################################################################
METRIC_TO_OPTIMISE = ['MulticlassAccuracy__2', 'MulticlassF1Score__2']  
LOSS = 'cross_entropy' 
PROBLEM = 'classification'
DATASET_NAME = "imdb"

INITIAL_MODEL = 'arnir0/Tiny-LLM'

model = AutoModelForSequenceClassification.from_pretrained(INITIAL_MODEL, num_labels=2, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(INITIAL_MODEL)
tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
model.config.pad_token_id = tokenizer.pad_token_id

INITIAL_ASSUMPTION = model

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors=None
    )
    if "label" in examples:
            tokenized["labels"] = examples["label"]
    return tokenized

class TextClassificationDataset(torch.utils.data.Dataset):
    """Dataset для классификации текста (возвращает кортеж для совместимости с convert_callable_loader)"""
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        
        labels = item['labels']
        if isinstance(labels, (list, torch.Tensor)):
            labels = int(labels[0]) if isinstance(labels, list) else int(labels.item())
        else:
            labels = int(labels)
        
        return (
            torch.tensor(item['input_ids'], dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(item['attention_mask'], dtype=torch.long)
        )


def collate_fn(batch):
    """
    Custom collate function that handles batching for LLM data.
    Converts tuple format (input_ids, labels, attention_mask) to stacked tensors.
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


def load_benchmark_dataset(dataset_name='imdb', batch_size=4, num_samples=None):
    """Load and prepare IMDb dataset for text classification"""
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
        eval_dataset = eval_dataset.select(range(min(num_samples//10, len(eval_dataset))))

    train_dataset = train_dataset.shuffle(seed=42)
    eval_dataset = eval_dataset.shuffle(seed=42)

    train_dataloader = DataLoader(
        TextClassificationDataset(train_dataset),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        TextClassificationDataset(eval_dataset),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        TextClassificationDataset(tokenized_datasets["test"]),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    fedcore_train_data = CompressionInputData(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        task=FedotTaskEnum.classification.value,
        num_classes=2,  
        input_dim=tokenizer.vocab_size,
    )
    
    fedcore_test_data = CompressionInputData(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        task=FedotTaskEnum.classification.value,
        num_classes=2,
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
                "num_reference_batches": 200,
            }
        )

        trained_model = trainer.fit(fedcore_train_data)

        ref_loss_history = trainer.history.get('ref_loss', [])

        train_loss_by_step = {}
        val_loss_by_step = {}

        internal_trainer = trainer._trainer if hasattr(trainer, '_trainer') else trainer.trainer_objects.get('trainer')

        if internal_trainer and hasattr(internal_trainer, 'state') and hasattr(internal_trainer.state, 'log_history'):
            log_history = internal_trainer.state.log_history
            
            # Диагностика: проверяем структуру первых записей
            if log_history:
                logger.info(f"First log entry keys: {list(log_history[0].keys())}")
                logger.info(f"Sample entries with step 500:")
                for entry in log_history:
                    if entry.get('step') == 500:
                        logger.info(f"  Step 500 entry: {entry}")
            
            for log_entry in log_history:
                step = log_entry.get('step', None)
                
                if step is not None:
                    step = int(step)  # Убеждаемся, что step - целое число
                    
                    # Training loss (может быть в записи без eval_loss или вместе с eval_loss)
                    if 'loss' in log_entry and 'eval_loss' not in log_entry:
                        # Запись только с training loss
                        train_loss_by_step[step] = _to_scalar(log_entry['loss'])
                    elif 'loss' in log_entry and 'eval_loss' in log_entry:
                        # Запись с обоими (на шагах оценки)
                        train_loss_by_step[step] = _to_scalar(log_entry['loss'])
                        val_loss_by_step[step] = _to_scalar(log_entry['eval_loss'])
                    
                    # Validation loss (может быть в отдельной записи или вместе с loss)
                    if 'eval_loss' in log_entry:
                        val_loss_by_step[step] = _to_scalar(log_entry['eval_loss'])

        logger.info(f"Ref loss history: {len(ref_loss_history)} entries")
        logger.info(f"Ref loss steps: {[s for s, _ in ref_loss_history][:5]}")  # Первые 5 шагов
        logger.info(f"Train loss entries: {len(train_loss_by_step)} entries")
        logger.info(f"Train loss steps: {sorted(train_loss_by_step.keys())[:5]}")  # Первые 5 шагов
        logger.info(f"Val loss entries: {len(val_loss_by_step)} entries")
        logger.info(f"Val loss steps: {sorted(val_loss_by_step.keys())[:5]}")  # Первые 5 шагов

        # Собираем все уникальные шаги
        all_steps = set()
        all_steps.update([int(step) for step, _ in ref_loss_history])  # Убеждаемся, что step - int
        all_steps.update(train_loss_by_step.keys())
        all_steps.update(val_loss_by_step.keys())
        all_steps = sorted(all_steps)

        results_data = []
        for step in all_steps:
            row = {'step': step}
            
            # ref_loss
            ref_loss_value = next((loss for s, loss in ref_loss_history if int(s) == step), None)
            row['ref_loss'] = _to_scalar(ref_loss_value) if ref_loss_value is not None else None
            
            # train_loss и val_loss
            row['train_loss'] = train_loss_by_step.get(step)
            row['val_loss'] = val_loss_by_step.get(step)
            
            results_data.append(row)

        df = pd.DataFrame(results_data)
        csv_path = os.path.join(results_dir, 'no_prune_res.csv')
        csv_path_abs = os.path.abspath(csv_path)
        df.to_csv(csv_path_abs, index=False)
        logger.info(f"Results saved to {csv_path_abs}")
        logger.info(f"CSV shape: {df.shape}")
        logger.info(f"Non-null train_loss: {df['train_loss'].notna().sum()}")
        logger.info(f"Non-null val_loss: {df['val_loss'].notna().sum()}")
        logger.info(f"Sample rows with all metrics:")
        # Показываем первые несколько строк, где есть все три метрики
        sample_rows = df[df[['ref_loss', 'train_loss', 'val_loss']].notna().all(axis=1)].head(3)
        logger.info(f"\n{sample_rows.to_string()}")