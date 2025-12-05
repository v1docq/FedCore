import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from fedot.core.repository.tasks import (
    Task,
    TaskTypesEnum,
)

from fedcore.api.config_factory import ConfigFactory
from fedcore.api.api_configs import (APIConfigTemplate, AutoMLConfigTemplate, FedotConfigTemplate,
                                     LearningConfigTemplate, DeviceConfigTemplate, ComputeConfigTemplate,
                                     LowRankTemplate)
from datasets import load_dataset
from fedcore.api.main import FedCore
from fedcore.api.llm_config import LLMConfigTemplate
from fedcore.data.data import CompressionInputData

##########################################################################
### CONFIGURATION ###
##########################################################################

INITIAL_MODEL = 'arnir0/Tiny-LLM'
model = AutoModelForCausalLM.from_pretrained(INITIAL_MODEL)
print(f"Model type: {type(INITIAL_MODEL)}")

TOKENIZER = AutoTokenizer.from_pretrained(INITIAL_MODEL)

if TOKENIZER.pad_token is None:
    if TOKENIZER.eos_token is not None:
        TOKENIZER.pad_token = TOKENIZER.eos_token
    else:
        TOKENIZER.add_special_tokens({'pad_token': '[PAD]'})
print(f"Tokenizer loaded: {TOKENIZER is not None}")
print(f"Vocabulary size: {TOKENIZER.vocab_size}")


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
        max_length=64,
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True)

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

wikitext_dataset = WikitextDataset(tokenized_dataset)

train_size = int(0.8 * len(wikitext_dataset))
val_size = int(0.1 * len(wikitext_dataset))
test_size = len(wikitext_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    wikitext_dataset, [train_size, val_size, test_size]
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

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)


example_batch = next(iter(val_dataloader))
example_input = example_batch[0]  


compression_data = CompressionInputData(
    features=example_input,  
    target=model, 
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    test_dataloader=test_dataloader,
    task=Task(TaskTypesEnum.classification),  
    input_dim=TOKENIZER.vocab_size,
)

################################################################################
### CONFIGURE FEDCORE WITH LLMTrainer AND LOW_RANK PEFT ###
################################################################################

pretrain_config = LLMConfigTemplate(
    epochs=5,  
    optimizer='paralleltg',
    scheduler='one_cycle',
    scheduler_step_each=1,
    criterion='cross_entropy',
    is_llm=True,
    model=model,
    tokenizer=TOKENIZER,
    custom_learning_params={
        'batch_size': 4,
        'learning_rate': 5e-5,
        'warmup_steps': 50,
        'logging_steps': 25,
        'save_steps': 100
    }
)

# finetune_config = LLMConfigTemplate(
#     model_architecture=model_arch_config,
#     epochs=2,
#     optimizer='ultg',
#     criterion=LOSS,
#     is_llm=True,
#     model=INITIAL_ASSUMPTION,
#     tokenizer=TOKENIZER,
#     custom_learning_params={
#         'batch_size': 4,
#         'learning_rate': 1e-5
#     }
# )

peft_config = LowRankTemplate(
    strategy='quantile',
    rank_prune_each=1, 
    custom_criterions=None,
    non_adaptive_threshold=0.3,  
    epochs=5,
    log_each=1,
    eval_each=1,
    decomposer='rsvd', 
    rank=None,  
    distortion_factor=0.6, 
    random_init='normal',  
    power=3,
)

fedot_config = FedotConfigTemplate(
    problem='classification',
    metric= ['accuracy', 'f1'],
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
    output_folder='./llm_low_rank_experiment',
    automl_folder='./llm_low_rank_automl'
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
    if hasattr(fedcore_compressor, 'fedcore_model'):
        model_class = fedcore_compressor.fedcore_model.__class__.__name__
        print(f"Trainer: {model_class}")

        if hasattr(fedcore_compressor.fedcore_model, 'operation_impl'):
            trainer_type = type(fedcore_compressor.fedcore_model.operation_impl).__name__
            print(f"Trainer type: {trainer_type}")
    model_comparison = fedcore_compressor.get_report(compression_data)