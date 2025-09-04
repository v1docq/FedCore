#!/usr/bin/env python3
"""
Test: training small transformer using GaLoreAdamW on wikitext-2 through LLMTrainer
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from galore_torch import GaLoreAdamW
from fedcore.models.network_impl.llm_trainer import LLMTrainer
import torch

# 1. Loading Model and Tokenizer
model_name = "sshleifer/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. Dataset Loading
print("Loading dataset wikitext-2...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Debug: looking at raw data
print("Raw data examples:")
for i in range(5):
    print(f"Sample {i}: {repr(dataset['train'][i]['text'][:100])}")
print(f"Number of samples in train: {len(dataset['train'])}")

def tokenize_function(examples):
    processed_texts = []
    for text in examples["text"]:
        if text and len(text.strip()) > 10:
            processed_texts.append(text)
        else:
            processed_texts.append("This is a sample text for training.")
    
    tokens = tokenizer(processed_texts, truncation=True, padding="max_length", max_length=32)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_datasets = dataset.map(tokenize_function, batched=True)
print("Features after tokenization:", tokenized_datasets["train"].features)
print("First tokenized sample:", tokenized_datasets["train"][0])

# Debug: looking at tokenized samples
print("\nTokenized examples:")
for i in range(3):
    example = tokenized_datasets["train"][i]
    print(f"Example {i}:")
    print(f"  input_ids: {example['input_ids'][:10]}...")
    print(f"  attention_mask: {example['attention_mask'][:10]}...")
    print(f"  labels: {example['labels'][:10]}...")
    print()

train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(64))
eval_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(16))  # Добавляем eval dataset

# 3. GaLoreAdamW Initializer

def galore_optimizer_init(model):
    print("Creating GaLoreAdamW...")
    return GaLoreAdamW(model.parameters(), lr=0.01)  # Используем тот же learning rate

# 4. Training args
training_args = {
    "output_dir": "./test_llm_galore",
    "num_train_epochs": 20,
    "per_device_train_batch_size": 4,
    "logging_steps": 5,
    "save_steps": 20,
    "eval_steps": 20,
    "evaluation_strategy": "steps",
    "save_strategy": "steps",
    "learning_rate": 0.01,
    "lr_scheduler_type": "cosine",
    "custom_optimizer": galore_optimizer_init,
    "load_best_model_at_end": True,
}

# 5. Creating LLMTrainer
trainer = LLMTrainer(
    model,
    training_args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# 6. Training
print("Perform training...")
trainer._init_hooks()
trainer.fit(train_dataset)

# 7. Checking optimizer type
opt = trainer._transformers_trainer.optimizer
print("Optimizer type:", type(opt))

if hasattr(opt, 'optimizer'):
    inner_opt = opt.optimizer
    print("Inner optimizer type:", type(inner_opt))
    assert "galore_torch" in str(type(inner_opt)), "Inner optimizer is not from galore_torch!"
    print("✅ GaLoreAdamW is used as inner optimizer!")
else:
    assert "galore_torch" in str(type(opt)), "Optimizer is not from galore_torch!"
    print("✅ GaLoreAdamW is used as optimizer!")

# 8. Checking params
if hasattr(opt, 'optimizer'):
    opt = opt.optimizer
print("Optimizer lr:", opt.param_groups[0]["lr"])
print("Expected initial lr was 0.001, current lr may be different due to scheduler")
print("✅ Learning rate is accessible and optimizer is working!") 