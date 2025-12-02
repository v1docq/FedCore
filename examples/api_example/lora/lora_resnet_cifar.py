"""
LoRA Training Example using FedCore API.

This example demonstrates how to use LoRA for parameter-efficient fine-tuning
through the FedCore API interface.

Task: Image classification on CIFAR-10
Model: ResNet18
Method: LoRA (Low-Rank Adaptation)
"""

from fedcore.api.config_factory import ConfigFactory
from fedcore.api.api_configs import (
    APIConfigTemplate, 
    AutoMLConfigTemplate, 
    FedotConfigTemplate,
    LearningConfigTemplate, 
    LoRATemplate
)
from fedcore.data.dataloader import load_data
from fedcore.api.main import FedCore


##########################################################################
# CONFIGURATION
##########################################################################

# Task and metrics
METRIC_TO_OPTIMISE = ['accuracy', 'latency']
LOSS = 'cross_entropy'
PROBLEM = 'classification'
PEFT_PROBLEM = 'lora_training'  # Use LoRA strategy

# Model configuration
INITIAL_ASSUMPTION = {
    #'path_to_model': 'pretrain_models/pretrain_model_checkpoint_at_15_epoch.pt',
    'path_to_model': 'C:/Users/Bars/PycharmProjects/FedCore/examples/api_example/pruning/cv_task/pretrain_models/pretrain_model_checkpoint_at_15_epoch.pt',
    'model_type': 'ResNet18'
}
INITIAL_MODEL = 'ResNet18'
PRETRAIN_SCENARIO = 'from_checkpoint'

# Optimization parameters
POP_SIZE = 1  # Single model (no evolutionary optimization)
TIMEOUT = 1   # Quick run

# Dataset
DATASET = 'CIFAR10'
train_dataloader_params = {
    "batch_size": 64,
    'shuffle': True,
    'is_train': True,
    'data_type': 'table',
    'split_ratio': [0.8, 0.2]
}
test_dataloader_params = {
    "batch_size": 100,
    'shuffle': True,
    'is_train': False,
    'data_type': 'table'
}


##########################################################################
# LORA CONFIGURATION
##########################################################################

# LoRA-specific parameters using LoRATemplate
lora_config = LoRATemplate(
    # Training parameters
    epochs=3,              # Number of training epochs
    log_each=1,             # Log every epoch
    eval_each=2,            # Evaluate every 2 epochs
    
    # LoRA-specific parameters
    lora_r=8,                        # LoRA rank
    lora_alpha=16,                   # LoRA scaling factor
    lora_dropout=0.1,                # Dropout for LoRA layers
    lora_target_modules=['layer4', 'fc'],  # Target layers
    use_peft=False,                  # Use FedCore implementation
    lora_bias='none',                # Bias strategy
)


##########################################################################
# API SETUP
##########################################################################

# FEDOT configuration (evolutionary optimization)
fedot_config = FedotConfigTemplate(
    problem=PROBLEM,
    metric=METRIC_TO_OPTIMISE,
    pop_size=POP_SIZE,
    timeout=TIMEOUT,
    initial_assumption=INITIAL_ASSUMPTION
)

# AutoML configuration
automl_config = AutoMLConfigTemplate(fedot_config=fedot_config)

# Learning configuration with LoRA
learning_config = LearningConfigTemplate(
    criterion=LOSS,
    learning_strategy=PRETRAIN_SCENARIO,
    peft_strategy=PEFT_PROBLEM,        # 'lora_training'
    peft_strategy_params=lora_config,
    exclude_lora=False                 # Include LoRA in optimization
)

# Complete API template
api_template = APIConfigTemplate(
    automl_config=automl_config,
    learning_config=learning_config
)


##########################################################################
# DATA LOADING
##########################################################################

def load_benchmark_dataset(dataset_name, train_params, test_params):
    """Load dataset using FedCore data loader."""
    fedcore_train_data = load_data(source=dataset_name, loader_params=train_params)
    fedcore_test_data = load_data(source=dataset_name, loader_params=test_params)
    return fedcore_train_data, fedcore_test_data


##########################################################################
# MAIN EXECUTION
##########################################################################

def main():
    """Main execution function."""
    print("="*80)
    print("LoRA Training Example - FedCore API")
    print("="*80)
    
    # Step 1: Create API configuration
    print("\n[Step 1] Creating API configuration...")
    APIConfig = ConfigFactory.from_template(api_template)
    api_config = APIConfig()
    print(f"  PEFT strategy: {PEFT_PROBLEM}")
    print(f"  LoRA rank: 8")
    print(f"  LoRA alpha: 16")
    print(f"  Target modules: ['layer4', 'fc']")
    
    # Step 2: Create FedCore instance
    print("\n[Step 2] Creating FedCore instance...")
    fedcore_compressor = FedCore(api_config)
    print("  FedCore API initialized")
    
    # Step 3: Load dataset
    print("\n[Step 3] Loading CIFAR-10 dataset...")
    fedcore_train_data, fedcore_test_data = load_benchmark_dataset(
        DATASET, 
        train_dataloader_params,
        test_dataloader_params
    )
    print(f"  Dataset loaded: {DATASET}")
    print(f"  Train/test split configured")
    
    # Step 4: Train model with LoRA
    print("\n[Step 4] Training ResNet18 with LoRA...")
    print("  This will:")
    print("    1. Load pre-trained ResNet18")
    print("    2. Apply LoRA adapters to specified layers")
    print("    3. Freeze base model parameters")
    print("    4. Train only LoRA parameters")
    
    fedcore_compressor.fit(fedcore_train_data)
    
    # Step 5: Access models
    print("\n[Step 5] Accessing trained models...")
    
    # Extract actual model from pipeline
    fitted_operation = fedcore_compressor.fedcore_model.root_node.fitted_operation
    
    if hasattr(fitted_operation, 'model_after'):
        lora_model = fitted_operation.model_after
        original_model = fitted_operation.model_before
        
        print(f"  Original model: {type(original_model).__name__}")
        print(f"  LoRA-adapted model: {type(lora_model).__name__}")
        
        # Count parameters
        if lora_model is not None:
            trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in lora_model.parameters())
            print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    else:
        print("  Model extraction: fitted_operation found")
    
    # Step 6: Evaluate and compare
    print("\n[Step 6] Evaluating models...")
    model_comparison = fedcore_compressor.get_report(fedcore_test_data)
    
    print("\n" + "="*80)
    print("Quality Metrics Comparison:")
    print("="*80)
    print(model_comparison['quality_comparison'])
    
    print("\n" + "="*80)
    print("Computational Metrics Comparison:")
    print("="*80)
    print(model_comparison['computational_comparison'])
    
    print("\n" + "="*80)
    print("LoRA Training Complete!")
    print("="*80)
    print("\nKey Results:")
    if 'lora_model' in locals() and lora_model is not None:
        print(f"  - {100*trainable/total:.2f}% of parameters trained")
        print(f"  - Storage: {100*trainable/total:.2f}% of full model checkpoint")


if __name__ == "__main__":
    main()

