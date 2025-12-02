"""
LoRA on HuggingFace Vision Transformer using FedCore API.

This example demonstrates LoRA on Vision Transformer for image classification.

Task: Image classification on CIFAR-10
Model: google/vit-base-patch16-224-in21k (Vision Transformer)
Method: LoRA on attention Query and Value projections
"""

import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from fedcore.api.config_factory import ConfigFactory
from fedcore.api.api_configs import (
    APIConfigTemplate, 
    AutoMLConfigTemplate, 
    FedotConfigTemplate,
    LearningConfigTemplate, 
    LoRATemplate
)
from fedcore.data.dataloader import load_data
from fedcore.data.data import CompressionInputData
from fedcore.api.main import FedCore
from fedot.core.repository.tasks import Task, TaskTypesEnum
import torchvision.transforms as transforms


##########################################################################
# CONFIGURATION
##########################################################################

METRIC_TO_OPTIMISE = ['accuracy']
LOSS = 'cross_entropy'
PROBLEM = 'classification'
PEFT_PROBLEM = 'lora_training'

# Vision Transformer configuration
MODEL_NAME = 'google/vit-base-patch16-224-in21k'
print(f"Loading Vision Transformer: {MODEL_NAME}")

# Load model directly
vit_model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=10,  # CIFAR-10 classes
    ignore_mismatched_sizes=True
)
print(f"Model loaded: {type(vit_model).__name__}")
print(f"Total parameters: {sum(p.numel() for p in vit_model.parameters()):,}")

POP_SIZE = 1
TIMEOUT = 2
DATASET = 'CIFAR10'

train_dataloader_params = {
    "batch_size": 16,
    'shuffle': True,
    'is_train': True,
    'data_type': 'table',
    'split_ratio': [0.8, 0.2],
    'resize': (224, 224)  # ViT needs 224x224
}

test_dataloader_params = {
    "batch_size": 32,
    'shuffle': False,
    'is_train': False,
    'data_type': 'table',
    'resize': (224, 224)
}


##########################################################################
# LORA CONFIGURATION FOR TRANSFORMERS
##########################################################################

lora_config = LoRATemplate(
    epochs=3,
    log_each=1,
    eval_each=1,
    
    # LoRA parameters for attention layers
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    lora_target_modules=['query', 'value'],  # Attention Q, V projections
    use_peft=False,  # Use FedCore implementation
    lora_bias='none',
)


##########################################################################
# API SETUP
##########################################################################

# Pass model directly in initial_assumption
INITIAL_ASSUMPTION = vit_model  # PyTorch model

fedot_config = FedotConfigTemplate(
    problem=PROBLEM,
    metric=METRIC_TO_OPTIMISE,
    pop_size=POP_SIZE,
    timeout=TIMEOUT,
    initial_assumption=INITIAL_ASSUMPTION
)

automl_config = AutoMLConfigTemplate(fedot_config=fedot_config)

learning_config = LearningConfigTemplate(
    criterion=LOSS,
    learning_strategy='from_scratch',
    peft_strategy=PEFT_PROBLEM,
    peft_strategy_params=lora_config,
    exclude_lora=False
)

api_template = APIConfigTemplate(
    automl_config=automl_config,
    learning_config=learning_config
)


##########################################################################
# MAIN EXECUTION
##########################################################################

def main():
    """Run LoRA training on Vision Transformer."""
    print("="*80)
    print("LoRA on Vision Transformer - FedCore API Example")
    print("="*80)
    
    print("\n[INFO] Configuration:")
    print(f"  Model: Vision Transformer (ViT)")
    print(f"  Dataset: {DATASET}")
    print(f"  LoRA rank: 8")
    print(f"  LoRA alpha: 16")
    print(f"  Target: Query & Value attention projections")
    print(f"  Epochs: 3")
    
    # Step 1: Create API configuration
    print("\n[Step 1] Initializing FedCore API...")
    APIConfig = ConfigFactory.from_template(api_template)
    api_config = APIConfig()
    fedcore_api = FedCore(api_config)
    print("  FedCore API initialized")
    
    # Step 2: Load dataset
    print("\n[Step 2] Loading CIFAR-10 dataset...")
    train_data = load_data(source=DATASET, loader_params=train_dataloader_params)
    test_data = load_data(source=DATASET, loader_params=test_dataloader_params)
    print(f"  Dataset loaded: {DATASET}")
    print(f"  Images resized to 224x224 for ViT")
    
    # Step 3: Train with LoRA
    print("\n[Step 3] Training Vision Transformer with LoRA...")
    print("  LoRA will be applied to Query and Value projections")
    print("  Only LoRA adapters will be trained (base model frozen)")
    
    fedcore_api.fit(train_data)
    
    # Step 4: Access models
    print("\n[Step 4] Accessing trained models...")
    
    fitted_operation = fedcore_api.fedcore_model.root_node.fitted_operation
    
    if hasattr(fitted_operation, 'model_after'):
        lora_model = fitted_operation.model_after
        original_model = fitted_operation.model_before
        
        print(f"  Original model: {type(original_model).__name__}")
        print(f"  LoRA-adapted model: {type(lora_model).__name__}")
        
        if lora_model is not None:
            trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in lora_model.parameters())
            print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    else:
        print("  Model extraction: fitted_operation found")
    
    # Step 5: Evaluate
    print("\n[Step 5] Evaluating models...")
    report = fedcore_api.get_report(test_data)
    
    print("\nQuality Metrics:")
    print(report['quality_comparison'])
    
    print("\nComputational Metrics:")
    print(report['computational_comparison'])
    
    print("\n" + "="*80)
    print("LoRA Training on Vision Transformer - COMPLETED")
    print("="*80)
    print("\nKey Results:")
    if 'lora_model' in locals() and lora_model is not None:
        print(f"  - {100*trainable/total:.2f}% of parameters trained")
        print(f"  - Memory efficient fine-tuning on Transformer!")
    
    return fedcore_api


if __name__ == "__main__":
    main()
