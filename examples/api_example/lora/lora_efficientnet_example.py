"""
LoRA on EfficientNet using FedCore API.

This example demonstrates LoRA on convolutional networks (EfficientNet)
for parameter-efficient fine-tuning through FedCore API.

Task: Image classification
Model: EfficientNet-B0
Method: LoRA on Conv2d layers
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
# CONFIGURATION FOR EFFICIENTNET + LORA
##########################################################################

METRIC_TO_OPTIMISE = ['accuracy', 'latency']
LOSS = 'cross_entropy'
PROBLEM = 'classification'
PEFT_PROBLEM = 'lora_training'

# EfficientNet configuration
INITIAL_ASSUMPTION = {
    'model_type': 'efficientnet_b0',
    'path_to_model': ''  # Empty for from_scratch
}
PRETRAIN_SCENARIO = 'from_scratch'

POP_SIZE = 1
TIMEOUT = 2
DATASET = 'CIFAR10'

train_dataloader_params = {
    "batch_size": 32,
    'shuffle': True,
    'is_train': True,
    'data_type': 'table',
    'split_ratio': [0.8, 0.2]
}

test_dataloader_params = {
    "batch_size": 64,
    'shuffle': False,
    'is_train': False,
    'data_type': 'table'
}


##########################################################################
# LORA CONFIGURATION FOR CONVOLUTIONAL LAYERS
##########################################################################

lora_config = LoRATemplate(
    epochs=5,
    log_each=1,
    eval_each=1,
    
    # LoRA parameters
    lora_r=8,                        # LoRA rank
    lora_alpha=16,                   # Scaling factor
    lora_dropout=0.1,                # Dropout
    lora_target_modules=[],          # Empty = apply to all Conv2d layers
    use_peft=False,                  # Use FedCore for Conv2d
    lora_bias='none',
)


##########################################################################
# API CONFIGURATION
##########################################################################

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
    learning_strategy=PRETRAIN_SCENARIO,
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
    """Run LoRA training on EfficientNet."""
    print("="*80)
    print("LoRA on EfficientNet-B0 - FedCore API Example")
    print("="*80)
    
    print("\n[INFO] Configuration:")
    print(f"  Model: EfficientNet-B0")
    print(f"  Dataset: {DATASET}")
    print(f"  LoRA rank: 8")
    print(f"  LoRA alpha: 16")
    print(f"  Target: All Conv2d layers (empty target_modules)")
    print(f"  Epochs: 5")
    
    # Create API
    print("\n[Step 1] Initializing FedCore API...")
    APIConfig = ConfigFactory.from_template(api_template)
    api_config = APIConfig()
    fedcore_api = FedCore(api_config)
    
    # Load data
    print("\n[Step 2] Loading CIFAR-10 dataset...")
    train_data = load_data(source=DATASET, loader_params=train_dataloader_params)
    test_data = load_data(source=DATASET, loader_params=test_dataloader_params)
    
    print(f"  Dataset loaded: {DATASET}")
    print(f"  Train/test split configured")
    
    # Train with LoRA
    print("\n[Step 3] Training EfficientNet with LoRA...")
    print("  LoRA will be applied to Conv2d layers automatically")
    print("  Only LoRA adapters will be trained (base model frozen)")
    
    fedcore_api.fit(train_data)
    
    # Get models
    print("\n[Step 4] Accessing models...")
    
    # Extract actual model from pipeline
    fitted_operation = fedcore_api.fedcore_model.root_node.fitted_operation
    
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
    
    # Evaluate
    print("\n[Step 5] Evaluating models...")
    report = fedcore_api.get_report(test_data)
    
    print("\nQuality Metrics:")
    print(report['quality_comparison'])
    
    print("\nComputational Metrics:")
    print(report['computational_comparison'])
    
    print("\n" + "="*80)
    print("LoRA Training on EfficientNet - COMPLETED")
    print("="*80)
    print("\nKey Results:")
    if 'lora_model' in locals() and lora_model is not None:
        print(f"  - {100*trainable/total:.2f}% of parameters trained")
        print(f"  - Storage: {100*trainable/total:.2f}% of full model checkpoint")
    
    return fedcore_api


if __name__ == "__main__":
    main()

