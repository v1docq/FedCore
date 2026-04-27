import sys
import os
os.environ["WANDB_DISABLED"] = "true"
import torch
import time
import gc
import logging
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from types import SimpleNamespace
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import torch_pruning as tp
# Правильный путь с учетом вложенности
correct_path = "/home/user/projects/FedCore/FedCore"
sys.path.insert(0, correct_path)
sys.path.insert(0, "/home/user/projects/FedCore/FedCore/AdaptPruner")
from AdaptPruner.LLMPruner.pruner.hf_pruner import (
    HFRMSNormPrunner,
    HFAttentionPrunner,
    HFLinearPrunner,
    MagnitudeImportance,
    TaylorImportance,
    hf_attention_pruner,
    hf_rmsnorm_pruner,
    hf_linear_pruner
)
from datasets import load_dataset


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
METRIC_TO_OPTIMISE = ['Perplexity']  
LOSS = 'cross_entropy' 
PROBLEM = 'classification'
DATASET_NAME = "imdb"

# INITIAL_MODEL = 'Qwen/Qwen1.5-MoE-A2.7B'
INITIAL_MODEL = 'Qwen/Qwen2.5-0.5B'
device_cpu = torch.device("cpu")
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    INITIAL_MODEL,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
model = model.to(device_cpu) 
tokenizer = AutoTokenizer.from_pretrained(INITIAL_MODEL)
tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
model.config.pad_token_id = tokenizer.pad_token_id

# model.tie_weights()
# model.lm_head.weight = model.model.embed_tokens.weight
print(model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr())

INITIAL_ASSUMPTION = model

# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

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

def apply_adapt_pruner_main(model, tokenizer, target_ratio=0.1, device="cuda"):
    import torch
    import os
    from types import SimpleNamespace
    from AdaptPruner.utils.hf_prune import main as adapt_prune_main

    tmp_dir = "tmp_pruning"
    os.makedirs(tmp_dir, exist_ok=True)

    tmp_model_path = os.path.join(tmp_dir, "model.pt")
    tmp_output_path = os.path.join(tmp_dir, "pruned_model.pt")

    torch.save(
        {"model": model, "tokenizer": tokenizer},
        tmp_model_path
    )

    total_params = sum(p.numel() for p in model.parameters())
    target_param_num = int(total_params * target_ratio)

    pruning_ratio = 1 - target_ratio
    # print("Original params:", total_params)
    # print("Target params:", target_param_num)

    args = SimpleNamespace(
        base_model=tmp_model_path,
        save_log_name="adapt_prune_exp",
        output_pth=tmp_output_path,

        target_param_num=target_param_num,
        pruning_ratio=pruning_ratio,
        pruner_type="taylor",

        block_wise=True,

        block_attention_layer_start=0,
        block_attention_layer_end=len(model.model.layers),

        block_mlp_layer_start=0,
        block_mlp_layer_end=len(model.model.layers),

        layer_wise=False,

        iterative_steps=50,

        grouping_strategy="sum",

        calibration_data_path="slimpajama",

        taylor="param_first",

        num_examples=64,
        taylor_seq_len=64,
        batch_size=8,

        adpative_prune=True,
        layer_prune_distribution_amplitude=0.5,
        layer_imp_method="cosine",

        device=device,
        seed=42,
        save_model=True,
        torch_version=float('.'.join(torch.__version__.split('.')[:2]))
    )

    adapt_prune_main(args)

    pruned_dict = torch.load(tmp_output_path, map_location=device)

    return pruned_dict["model"]


if __name__ == "__main__":
        fedcore_train_data, fedcore_test_data = load_benchmark_dataset('imdb', batch_size=4, num_samples=None)
        print(model)
        # pruned_model = apply_llmpruner_pruning(model, tokenizer, amount=0.5)
        model.gradient_checkpointing_disable()
        model.config.use_cache = False
        pruned_model = apply_adapt_pruner_main(
            model,
            tokenizer,
            target_ratio=0.1
        )
        print(pruned_model)
        def count_params(model):
            return sum(p.numel() for p in model.parameters())

        print("Params after pruning:", count_params(pruned_model))
        torch.save(pruned_model.state_dict(), "qwen_pruned_cpu.pth")
        pruned_model = pruned_model.to(device_gpu)
        trainer = LLMTrainer(
            model=pruned_model,
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
        torch.save(trained_model.state_dict(), "qwen_pruned_trained.pth")

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

        logger.info(f"Ref loss history: {len(ref_loss_history)} entries")
        logger.info(f"Ref loss steps: {[s for s, _ in ref_loss_history][:5]}")  
        logger.info(f"Train loss entries: {len(train_loss_by_step)} entries")
        logger.info(f"Train loss steps: {sorted(train_loss_by_step.keys())[:5]}")  
        logger.info(f"Val loss entries: {len(val_loss_by_step)} entries")
        logger.info(f"Val loss steps: {sorted(val_loss_by_step.keys())[:5]}")  

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
        csv_path = os.path.join(results_dir, 'prune_090.csv')
        csv_path_abs = os.path.abspath(csv_path)
        df.to_csv(csv_path_abs, index=False)
        logger.info(f"Results saved to {csv_path_abs}")
        logger.info(f"CSV shape: {df.shape}")
        logger.info(f"Non-null train_loss: {df['train_loss'].notna().sum()}")
        logger.info(f"Non-null val_loss: {df['val_loss'].notna().sum()}")
        logger.info(f"Sample rows with all metrics:")
        sample_rows = df[df[['ref_loss', 'train_loss', 'val_loss']].notna().all(axis=1)].head(3)
        logger.info(f"\n{sample_rows.to_string()}")