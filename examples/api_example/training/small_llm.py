import sys
import os
import torch

# Правильный путь с учетом вложенности
correct_path = "/home/user/projects/FedCore/FedCore"
sys.path.insert(0, correct_path)

print("Обновленный sys.path:")
print(sys.path[0])
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from fedot.core.repository.tasks import (
    Task,
    TaskTypesEnum,
)

from fedcore.api.config_factory import ConfigFactory
from fedcore.api.api_configs import (APIConfigTemplate, AutoMLConfigTemplate, FedotConfigTemplate,
                                     LearningConfigTemplate, ModelArchitectureConfigTemplate,
                                     NeuralModelConfigTemplate, DeviceConfigTemplate, ComputeConfigTemplate)
from fedcore.architecture.dataset.api_loader import ApiLoader
from fedcore.data.dataloader import load_data
from datasets import load_dataset
from fedcore.tools.example_utils import get_scenario_for_api
from fedcore.api.main import FedCore
from fedcore.api.llm_config import LLMConfigTemplate
from fedcore.data.data import CompressionInputData
from fedcore.repository.constanst_repository import FedotTaskEnum

##########################################################################
### DEFINE ML PROBLEM (classification, object_detection, regression,   ###
### ts_forecasting, question_answering, summarization), PEFT problem   ###
### (pruning, quantisation, distillation,                              ###
### low_rank, training) and appropriate loss function both for model   ###
### and compute                                                        ###
##########################################################################

METRIC_TO_OPTIMISE = ['perplexity', 'latency']
LOSS = 'cross_entropy'
PROBLEM = 'classification'
PEFT_PROBLEM = 'training'
INITIAL_MODEL = 'arnir0/Tiny-LLM'
INITIAL_ASSUMPTION = AutoModelForCausalLM.from_pretrained(INITIAL_MODEL)
print(f"Type: {INITIAL_ASSUMPTION.model_tags}")

TOKENIZER = AutoTokenizer.from_pretrained(INITIAL_MODEL)

if TOKENIZER.pad_token is None:
    if TOKENIZER.eos_token is not None:
        TOKENIZER.pad_token = TOKENIZER.eos_token
    else:
        TOKENIZER.add_special_tokens({'pad_token': '[PAD]'})
print(f"Tokenizer loaded: {TOKENIZER is not None}")
print(f"Vocabulary size: {TOKENIZER.vocab_size}")

PRETRAIN_SCENARIO = 'from_checkpoint'
SCRATCH_SCENARIO = 'from_scratch'
POP_SIZE = 1
DATASET = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")

def preprocess_dataset(examples):
    """Extract text from different dataset formats"""
    if 'text' in examples:
        return {"text": examples["text"]}
    else:
        for key, value in examples.items():
            if isinstance(value[0], str):
                return {"text": examples[key]}
        return {"text": [str(x) for x in examples[list(examples.keys())[0]]]}

dataset = DATASET.map(preprocess_dataset, batched=True)
dataset = dataset.filter(lambda example: len(example["text"].strip()) > 0)

def tokenize_function(examples):
    tokenized = TOKENIZER(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True)

class WikitextDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset
        
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        
        def safe_flatten(data):
            if isinstance(data, torch.Tensor):
                return data.flatten()
            elif isinstance(data, list):
                return torch.tensor(data).flatten()
            else:
                return torch.tensor([data]).flatten()
        
        return {
            'input_ids': safe_flatten(item['input_ids']),
            'attention_mask': safe_flatten(item['attention_mask']),
            'labels': safe_flatten(item['labels'])
        }

wikitext_dataset = WikitextDataset(tokenized_dataset)

train_size = int(0.8 * len(wikitext_dataset))
val_size = int(0.1 * len(wikitext_dataset))
test_size = len(wikitext_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    wikitext_dataset, [train_size, val_size, test_size]
)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

compression_data = CompressionInputData(
    features=None,  
    target=None,  
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    test_dataloader=test_dataloader,
    task=Task(TaskTypesEnum.classification),  
    input_dim=TOKENIZER.vocab_size, 
)

################################################################################
### CREATE SCENARIO FOR FEDCORE AGENT (TRAIN AND OPTIMISE MODEL FROM SCRATCH ###
### or optimise pretrained model with PEFT strategies                        ###
################################################################################
initial_assumption, learning_strategy = get_scenario_for_api(
    scenario_type=SCRATCH_SCENARIO,
    initial_assumption=INITIAL_ASSUMPTION
)

training_params = {
    'model': INITIAL_ASSUMPTION,
    'tokenizer': TOKENIZER,
    'is_llm': True,
    'epochs': 15,
    'optimizer': 'adam',
    'criterion': LOSS,
    'input_dim': TOKENIZER.vocab_size,
    'output_dim': TOKENIZER.vocab_size,
    'depth': 3
}

model_arch_config = ModelArchitectureConfigTemplate(
    input_dim=TOKENIZER.vocab_size,
    output_dim=TOKENIZER.vocab_size,
    depth=3
)

pretrain_config = NeuralModelConfigTemplate(
    model_architecture=model_arch_config,
    epochs=15,
    optimizer='adam',
    criterion=LOSS,
    custom_learning_params={
        'model': INITIAL_ASSUMPTION,
        'tokenizer': TOKENIZER,
        'is_llm': True,
        'batch_size': 4
    }
)

peft_neural_config = NeuralModelConfigTemplate(
    model_architecture=model_arch_config,
    epochs=15,
    optimizer='adam',
    criterion=LOSS,
    custom_learning_params={
        'model': INITIAL_ASSUMPTION,
        'tokenizer': TOKENIZER,
        'is_llm': True,
        'batch_size': 4
    }
)

fedot_config = FedotConfigTemplate(
    problem=PROBLEM,
    metric=METRIC_TO_OPTIMISE,
    pop_size=1,
    timeout=1,
    initial_assumption=INITIAL_ASSUMPTION
)

automl_config = AutoMLConfigTemplate(fedot_config=fedot_config)

learning_config = LearningConfigTemplate(
    criterion=LOSS,
    learning_strategy=learning_strategy,
    learning_strategy_params=pretrain_config,
    peft_strategy=PEFT_PROBLEM,
    peft_strategy_params=peft_neural_config
)

device_config = DeviceConfigTemplate(device='cuda')
compute_config = ComputeConfigTemplate(
    output_folder='./current_experiment_folder',
    automl_folder='./current_automl_folder'
)

api_template = APIConfigTemplate(
    device_config=device_config,
    automl_config=automl_config,
    learning_config=learning_config,
    compute_config=compute_config
)

if __name__ == "__main__":
    APIConfig = ConfigFactory.from_template(api_template)
    api_config = APIConfig()
    fedcore_compressor = FedCore(api_config)
    fedcore_compressor.fit_no_evo(compression_data)
    model_comparison = fedcore_compressor.get_report(compression_data)