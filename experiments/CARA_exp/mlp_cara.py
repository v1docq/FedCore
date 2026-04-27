import sys
import os
import torch
import logging
import random
import pandas as pd
import bitsandbytes as bnb

from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset

correct_path = "/home/user/projects/FedCore/FedCore"
sys.path.insert(0, correct_path)

from external.caracore.cara import CaraLinear
from fedcore.data.data import CompressionInputData
from fedcore.models.network_impl.llm_trainer import LLMTrainer

_script_dir = os.path.dirname(os.path.abspath(__file__))

log_dir = os.path.join(_script_dir, "logs")
results_dir = os.path.join(_script_dir, "results")

os.makedirs(log_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(os.path.join(log_dir, "experiment.log"))
console_handler = logging.StreamHandler(sys.stdout)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

METRICS = [
    "Perplexity",
    "latency_total",
    "latency_per_moe_block",
    "latency_per_expert",
    "num_parameters",
    "expert_activation"
]
CARA_RANK = 16
INITIAL_MODEL = "Qwen/Qwen1.5-MoE-A2.7B"

model = AutoModelForCausalLM.from_pretrained(
    INITIAL_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

tokenizer = AutoTokenizer.from_pretrained(INITIAL_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.pad_token_id

for p in model.parameters():
    p.requires_grad = False

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_parameters": total,
        "trainable_parameters": trainable
    }

class ExpertActivationTracker:

    def __init__(self, save_steps=500):

        self.counts = defaultdict(int)
        self.history = []
        self.save_steps = save_steps

    def record(self, expert_ids):

        for e in expert_ids:
            self.counts[int(e)] += 1

    def save_step(self, step):
        for expert, count in self.counts.items():

            self.history.append({
                "step": step,
                "expert_id": expert,
                "activation_count": count
            })

        self.counts = defaultdict(int)

    def save_csv(self, path):
        df = pd.DataFrame(self.history)
        df.to_csv(path, index=False)

def register_expert_hooks(model, tracker):
    for name, module in model.named_modules():
        if "router" in name.lower():
            def hook(module, inputs, outputs):
                if isinstance(outputs, tuple):
                    router_logits = outputs[0]
                else:
                    router_logits = outputs
                expert_ids = torch.argmax(router_logits, dim=-1)
                tracker.record(expert_ids.flatten().tolist())
            module.register_forward_hook(hook)

def save_experts(model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for name, module in model.named_modules():
        if hasattr(module, "experts"):
            for i, expert in enumerate(module.experts):
                path = os.path.join(save_dir, f"{name}_expert_{i}.pth")
                torch.save(expert.state_dict(), path)

def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

class MixedDataset(IterableDataset):
    def __init__(self, mmlu, c4, mix_ratio=0.3):
        self.mmlu = mmlu
        self.c4 = c4
        self.mix_ratio = mix_ratio

    def __iter__(self):
        mmlu_iter = iter(self.mmlu)
        c4_iter = iter(self.c4)
        while True:
            if random.random() < self.mix_ratio:
                try:
                    yield next(mmlu_iter)
                except StopIteration:
                    mmlu_iter = iter(self.mmlu)
                    yield next(mmlu_iter)
            else:
                yield next(c4_iter)

def load_mmlu():
    mmlu = load_dataset("cais/mmlu", "all")
    def format_example(example):
        question = example["question"]
        choices = example["choices"]
        answer = example["answer"]
        text = question + "\n"
        for i, c in enumerate(choices):
            text += f"{chr(65+i)}. {c}\n"
        text += f"Answer: {chr(65+answer)}"
        return {"text": text}
    mmlu = mmlu.map(format_example)

    return mmlu


def load_c4():
    c4 = load_dataset("allenai/c4", "en", streaming=True)
    return c4["train"]

def load_benchmark_dataset(batch_size=4):
    mmlu = load_mmlu()
    c4 = load_c4()

    mmlu_train = mmlu["train"].map(tokenize_function, batched=True)
    mmlu_val = mmlu["validation"].map(tokenize_function, batched=True)
    mmlu_test = mmlu["test"].map(tokenize_function, batched=True)

    c4_tokenized = c4.map(tokenize_function)
    train_dataset = MixedDataset(mmlu_train, c4_tokenized)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    val_loader = DataLoader(mmlu_val, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(mmlu_test, batch_size=batch_size, num_workers=0)

    fedcore_train_data = CompressionInputData(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        task="text_generation",
        num_classes=tokenizer.vocab_size,
        input_dim=tokenizer.vocab_size,
    )

    fedcore_test_data = CompressionInputData(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        task="text_generation",
        num_classes=tokenizer.vocab_size,
        input_dim=tokenizer.vocab_size,
    )

    return fedcore_train_data, fedcore_test_data

if __name__ == "__main__":

    fedcore_train_data, fedcore_test_data = load_benchmark_dataset(batch_size=4)

    shared_adapters = {}

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            if any(x in name for x in ["gate_proj","up_proj","down_proj"]):
                layer_type = name.split(".")[-1]
                if layer_type not in shared_adapters:
                    shared_adapters[layer_type] = CaraLinear(module, rank=CARA_RANK)
                parent = model
                parts = name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], shared_adapters[layer_type])

    optimizer = bnb.optim.Adam8bit(
        model.parameters(),
        lr=2e-5,
        betas=(0.9, 0.95),
        weight_decay=0.01
    )

    expert_tracker = ExpertActivationTracker(save_steps=500)
    register_expert_hooks(model, expert_tracker)
    param_stats = count_parameters(model)
    logger.info(param_stats)

    trainer = LLMTrainer(
        model=model,
        optimizer=optimizer,
        params={
            "tokenizer": tokenizer,
            "bf16": True,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "output_dir": "./output",
            "logging_steps": 500,
            "eval_steps": 500,
            "save_steps": 500
        }
    )

    trained_model = trainer.fit(fedcore_train_data)

    expert_tracker.save_csv(os.path.join(results_dir, "expert_activation_frequency.csv"))
    save_experts(trained_model, os.path.join(results_dir, "experts"))
    param_df = pd.DataFrame([param_stats])
    param_df.to_csv(os.path.join(results_dir, "model_parameters.csv"), index=False)
