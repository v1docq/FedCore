"""
LoRA Training Examples for FedCore.

This file demonstrates how to use LoRA (Low-Rank Adaptation) in FedCore
for parameter-efficient fine-tuning of neural networks.

Examples:
1. EfficientNet with LoRA (Conv2d layers)
2. Transformer with LoRA (HuggingFace model, optional PEFT)
3. Custom model with LoRA (ResNet)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def example_efficientnet_lora():
    """
    Example 1: EfficientNet with LoRA
    
    Demonstrates:
    - Loading EfficientNet model
    - Applying LoRA to convolutional layers
    - Parameter-efficient training
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: EfficientNet with LoRA (Conv2d Support)")
    print("="*80)
    
    from torchvision.models import efficientnet_b0
    from fedcore.algorithm.low_rank.lora_operation import BaseLoRA
    
    # Step 1: Load pre-trained EfficientNet
    print("\n[Step 1] Loading EfficientNet-B0...")
    model = efficientnet_b0(pretrained=True)
    
    # Get original parameter count
    original_params = sum(p.numel() for p in model.parameters())
    print(f"Original parameters: {original_params:,}")
    
    # Step 2: Configure LoRA parameters
    print("\n[Step 2] Configuring LoRA parameters...")
    lora_config = {
        'lora_r': 8,                    # Rank of LoRA matrices
        'lora_alpha': 16,               # Scaling factor
        'lora_dropout': 0.1,            # Dropout for regularization
        'lora_target_modules': [],      # Empty = apply to all suitable layers
        'use_peft': False,              # Use fedcore implementation
        'lora_bias': 'none',            # Don't train bias
        'epochs': 5,
        'lr': 1e-4
    }
    
    print(f"LoRA config: r={lora_config['lora_r']}, alpha={lora_config['lora_alpha']}")
    
    # Step 3: Create BaseLoRA instance
    print("\n[Step 3] Creating BaseLoRA instance...")
    lora = BaseLoRA(params=lora_config)
    
    # Step 4: Apply LoRA to model
    print("\n[Step 4] Applying LoRA to model...")
    model_with_lora = lora._apply_lora_to_model(model, lora_config)
    
    # Step 5: Freeze non-LoRA parameters
    print("\n[Step 5] Freezing base parameters...")
    lora._freeze_non_lora_parameters(model_with_lora)
    
    # Check trainable parameters
    trainable_params = sum(p.numel() for p in model_with_lora.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model_with_lora.parameters())
    
    print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,}")
    print(f"Percentage trainable: {100*trainable_params/total_params:.2f}%")
    print(f"Memory efficiency: {100*(1-trainable_params/total_params):.2f}% saved")
    
    # Step 6: Test forward pass
    print("\n[Step 6] Testing forward pass...")
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model_with_lora(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print("\n[SUCCESS] EfficientNet with LoRA is ready for training!")
    print(f"Memory-efficient training: only {100*trainable_params/total_params:.2f}% parameters to train")
    
    return model_with_lora


def example_transformer_lora():
    """
    Example 2: Transformer with LoRA (HuggingFace model)
    
    Demonstrates:
    - Loading HuggingFace transformer
    - Applying LoRA with PEFT (optional)
    - Training on text data
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Transformer with LoRA (HuggingFace + PEFT)")
    print("="*80)
    
    try:
        from transformers import AutoModel, AutoTokenizer
        
        # Step 1: Load small transformer model
        print("\n[Step 1] Loading BERT-tiny model...")
        model_name = "prajjwal1/bert-tiny"
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        original_params = sum(p.numel() for p in model.parameters())
        print(f"Original parameters: {original_params:,}")
        
        # Step 2: Configure LoRA for transformers
        print("\n[Step 2] Configuring LoRA for transformer...")
        lora_config_peft = {
            'lora_r': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.1,
            'lora_target_modules': ['query', 'value'],  # Attention layers
            'use_peft': True,               # Try PEFT first
            'lora_bias': 'none',
            'task_type': 'seq_cls',         # Sequence classification
        }
        
        lora_config_fedcore = {
            'lora_r': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.1,
            'lora_target_modules': [],      # Apply to all suitable layers
            'use_peft': False,              # Use fedcore implementation
            'lora_bias': 'none',
        }
        
        # Choose which config to use
        use_peft = False  # Set to True if peft is installed
        lora_config = lora_config_peft if use_peft else lora_config_fedcore
        
        print(f"Using {'PEFT' if use_peft else 'FedCore'} implementation")
        print(f"Target modules: {lora_config['lora_target_modules']}")
        
        # Step 3: Apply LoRA
        print("\n[Step 3] Applying LoRA to transformer...")
        from fedcore.algorithm.low_rank.lora_operation import BaseLoRA
        
        lora = BaseLoRA(params=lora_config)
        model_with_lora = lora._apply_lora_to_model(model, lora_config)
        lora._freeze_non_lora_parameters(model_with_lora)
        
        # Check parameters
        trainable_params = sum(p.numel() for p in model_with_lora.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model_with_lora.parameters())
        
        print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,}")
        print(f"Percentage trainable: {100*trainable_params/total_params:.2f}%")
        
        # Step 4: Test with sample text
        print("\n[Step 4] Testing with sample text...")
        text = ["Hello, this is a test.", "Another test sentence."]
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model_with_lora(**inputs)
        
        print(f"Output shape: {outputs.last_hidden_state.shape}")
        print("\n[SUCCESS] Transformer with LoRA is ready!")
        
        return model_with_lora
        
    except ImportError as e:
        print(f"\n[SKIP] Transformers library not installed: {e}")
        print("Install with: pip install transformers")
        return None


def example_custom_model_lora():
    """
    Example 3: Custom ResNet with LoRA
    
    Demonstrates:
    - Loading custom PyTorch model (ResNet)
    - Applying LoRA to specific layers
    - Fine-tuning for new task
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Custom ResNet with LoRA")
    print("="*80)
    
    from torchvision.models import resnet18
    from fedcore.algorithm.low_rank.lora_operation import BaseLoRA
    
    # Step 1: Load pre-trained ResNet
    print("\n[Step 1] Loading ResNet18...")
    model = resnet18(pretrained=True)
    
    # Modify for different number of classes
    num_classes = 10  # e.g., CIFAR-10
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    original_params = sum(p.numel() for p in model.parameters())
    print(f"Original parameters: {original_params:,}")
    print(f"Output classes: {num_classes}")
    
    # Step 2: Configure LoRA for specific layers
    print("\n[Step 2] Configuring LoRA for ResNet...")
    lora_config = {
        'lora_r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'lora_target_modules': ['layer4', 'fc'],  # Only last layer and fc
        'use_peft': False,
        'lora_bias': 'none',
        'epochs': 10,
        'lr': 2e-4
    }
    
    print(f"Targeting modules: {lora_config['lora_target_modules']}")
    
    # Step 3: Apply LoRA
    print("\n[Step 3] Applying LoRA to ResNet...")
    lora = BaseLoRA(params=lora_config)
    model_with_lora = lora._apply_lora_to_model(model, lora_config)
    lora._freeze_non_lora_parameters(model_with_lora)
    
    # Check parameters
    trainable_params = sum(p.numel() for p in model_with_lora.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model_with_lora.parameters())
    
    print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,}")
    print(f"Percentage trainable: {100*trainable_params/total_params:.2f}%")
    
    # Step 4: Simulate training loop
    print("\n[Step 4] Simulating training loop...")
    model_with_lora.train()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model_with_lora.parameters()),
        lr=lora_config['lr']
    )
    criterion = nn.CrossEntropyLoss()
    
    # Dummy batch
    dummy_images = torch.randn(4, 3, 224, 224)
    dummy_labels = torch.randint(0, num_classes, (4,))
    
    # Training step
    optimizer.zero_grad()
    outputs = model_with_lora(dummy_images)
    loss = criterion(outputs, dummy_labels)
    loss.backward()
    optimizer.step()
    
    print(f"Training step completed. Loss: {loss.item():.4f}")
    
    # Step 5: Save LoRA parameters only
    print("\n[Step 5] Saving LoRA parameters...")
    lora.model_after = model_with_lora
    lora_state = lora.get_lora_state_dict()
    
    print(f"LoRA parameters to save: {len(lora_state)} tensors")
    total_lora_params = sum(v.numel() for v in lora_state.values())
    print(f"Total LoRA parameters: {total_lora_params:,}")
    print(f"Storage efficiency: {100*total_lora_params/total_params:.2f}% of full model")
    
    print("\n[SUCCESS] Custom ResNet with LoRA trained successfully!")
    print(f"Only need to save {100*total_lora_params/total_params:.2f}% of parameters")
    
    return model_with_lora, lora_state


def example_lora_comparison():
    """
    Example 4: Compare different LoRA ranks
    
    Demonstrates:
    - Impact of rank on parameter count
    - Trade-off between efficiency and capacity
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Comparing Different LoRA Ranks")
    print("="*80)
    
    from torchvision.models import resnet18
    from fedcore.algorithm.low_rank.lora_operation import BaseLoRA
    
    model = resnet18(pretrained=False)
    base_params = sum(p.numel() for p in model.parameters())
    
    print(f"\nBase model parameters: {base_params:,}")
    print("\nComparing different ranks:")
    print("-" * 80)
    print(f"{'Rank':<8} {'Trainable Params':<20} {'Percentage':<12} {'Recommendation':<20}")
    print("-" * 80)
    
    ranks = [2, 4, 8, 16, 32, 64]
    
    for r in ranks:
        model_copy = resnet18(pretrained=False)
        
        lora_config = {
            'lora_r': r,
            'lora_alpha': 2*r,
            'lora_dropout': 0.1,
            'lora_target_modules': ['fc'],
            'use_peft': False
        }
        
        lora = BaseLoRA(params=lora_config)
        model_lora = lora._apply_lora_to_model(model_copy, lora_config)
        lora._freeze_non_lora_parameters(model_lora)
        
        trainable = sum(p.numel() for p in model_lora.parameters() if p.requires_grad)
        percentage = 100 * trainable / base_params
        
        # Recommendation based on rank
        if r <= 4:
            rec = "Very efficient"
        elif r <= 16:
            rec = "Balanced"
        elif r <= 32:
            rec = "High capacity"
        else:
            rec = "Maximum capacity"
        
        print(f"{r:<8} {trainable:<20,} {percentage:<12.3f}% {rec:<20}")
    
    print("-" * 80)
    print("\nRecommendations:")
    print("  - Rank 4-8:   Good for most fine-tuning tasks")
    print("  - Rank 16-32: Complex domain adaptation")
    print("  - Rank 64+:   Approaching full fine-tuning")
    
    print("\n[SUCCESS] Rank comparison complete")


def example_lora_save_load():
    """
    Example 5: Saving and Loading LoRA Parameters
    
    Demonstrates:
    - Training model with LoRA
    - Saving only LoRA parameters (not full model)
    - Loading LoRA parameters into base model
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Saving and Loading LoRA Parameters")
    print("="*80)
    
    from torchvision.models import resnet18
    from fedcore.algorithm.low_rank.lora_operation import BaseLoRA
    
    # Step 1: Train model with LoRA
    print("\n[Step 1] Training model with LoRA...")
    model = resnet18(pretrained=True)
    
    lora_config = {
        'lora_r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.0,
        'lora_target_modules': ['fc'],
        'use_peft': False
    }
    
    lora = BaseLoRA(params=lora_config)
    model_with_lora = lora._apply_lora_to_model(model, lora_config)
    lora._freeze_non_lora_parameters(model_with_lora)
    lora.model_after = model_with_lora
    
    # Simulate training
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model_with_lora.parameters()),
        lr=1e-4
    )
    
    for i in range(3):
        dummy_input = torch.randn(2, 3, 224, 224)
        dummy_target = torch.randint(0, 1000, (2,))
        
        optimizer.zero_grad()
        output = model_with_lora(dummy_input)
        loss = nn.CrossEntropyLoss()(output, dummy_target)
        loss.backward()
        optimizer.step()
    
    print(f"Training complete. Final loss: {loss.item():.4f}")
    
    # Step 2: Save ONLY LoRA parameters
    print("\n[Step 2] Saving LoRA parameters...")
    lora_state = lora.get_lora_state_dict()
    
    save_path = "lora_resnet18_checkpoint.pt"
    torch.save(lora_state, save_path)
    
    lora_size = sum(v.numel() for v in lora_state.values())
    total_size = sum(p.numel() for p in model_with_lora.parameters())
    
    print(f"LoRA parameters saved: {len(lora_state)} tensors")
    print(f"Storage size: {lora_size:,} parameters ({100*lora_size/total_size:.2f}% of full model)")
    print(f"Saved to: {save_path}")
    
    # Step 3: Load LoRA into fresh model
    print("\n[Step 3] Loading LoRA into fresh base model...")
    fresh_model = resnet18(pretrained=True)
    
    fresh_lora = BaseLoRA(params=lora_config)
    fresh_model_lora = fresh_lora._apply_lora_to_model(fresh_model, lora_config)
    fresh_lora.model_after = fresh_model_lora
    
    # Load LoRA state
    loaded_state = torch.load(save_path)
    fresh_lora.load_lora_state_dict(loaded_state)
    
    print(f"Loaded {len(loaded_state)} LoRA parameter tensors")
    print("\n[SUCCESS] LoRA parameters saved and loaded successfully!")
    print(f"Storage savings: {100*(1-lora_size/total_size):.1f}%")
    
    return fresh_model_lora


def example_lora_multiple_models():
    """
    Example 6: Applying LoRA to Multiple Model Architectures
    
    Demonstrates:
    - LoRA on different architectures
    - Comparing parameter efficiency
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: LoRA on Multiple Architectures")
    print("="*80)
    
    from torchvision.models import resnet18, efficientnet_b0, mobilenet_v3_small
    from fedcore.algorithm.low_rank.lora_operation import BaseLoRA
    
    models = {
        'ResNet18': resnet18(pretrained=False),
        'EfficientNet-B0': efficientnet_b0(pretrained=False),
        'MobileNetV3-Small': mobilenet_v3_small(pretrained=False),
    }
    
    lora_config = {
        'lora_r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'lora_target_modules': [],
        'use_peft': False
    }
    
    print("\nApplying LoRA with rank=8 to different architectures:")
    print("-" * 80)
    print(f"{'Model':<20} {'Total Params':<15} {'Trainable':<15} {'Percentage':<12}")
    print("-" * 80)
    
    for model_name, model in models.items():
        lora = BaseLoRA(params=lora_config)
        model_lora = lora._apply_lora_to_model(model, lora_config)
        lora._freeze_non_lora_parameters(model_lora)
        
        total = sum(p.numel() for p in model_lora.parameters())
        trainable = sum(p.numel() for p in model_lora.parameters() if p.requires_grad)
        percentage = 100 * trainable / total
        
        print(f"{model_name:<20} {total:<15,} {trainable:<15,} {percentage:<12.2f}%")
    
    print("-" * 80)
    print("\n[SUCCESS] LoRA works across different architectures!")


def example_lora_with_api():
    """
    Example 7: Using LoRA through FedCore API (Conceptual)
    
    Note: This is a conceptual example showing how to use LoRA through API.
    Actual execution requires full FedCore/FEDOT environment.
    """
    print("\n" + "="*80)
    print("EXAMPLE 7: Using LoRA through FedCore API (Conceptual)")
    print("="*80)
    
    print("\n[Conceptual Code Example]")
    print("-" * 80)
    
    code_example = '''
# Import FedCore API (requires full environment)
from fedcore.api.main import ModelCompressionAPI

# Configure LoRA parameters
config = {
    'learning_config': {
        'peft_strategy': 'lora_training',
        'peft_strategy_params': {
            'lora_r': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.1,
            'lora_target_modules': ['layer4', 'fc'],
            'use_peft': False,
        },
        'exclude_lora': False  # Include LoRA in evolutionary optimization
    },
    'automl_config': {
        'problem': 'lora_training',
        'timeout': 10,
        'pop_size': 5
    }
}

# Create API instance
api = ModelCompressionAPI(config=config)

# Fit model with LoRA
api.fit(train_data)

# Get compressed model
compressed_model = api.predict(test_data)

# Access LoRA-adapted model
lora_model = api.solver.model_after
'''
    
    print(code_example)
    print("-" * 80)
    
    print("\n[Key Points]")
    print("  1. Use peft_strategy='lora_training' to enable LoRA")
    print("  2. Configure LoRA parameters via peft_strategy_params")
    print("  3. Set exclude_lora=True to exclude from evolutionary optimization")
    print("  4. Access trained model via api.solver.model_after")
    
    print("\n[INFO] This is a conceptual example")
    print("Actual usage requires installed fedot and full FedCore dependencies")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("LORA USAGE EXAMPLES - Stage 10")
    print("="*80)
    print("\nThis script demonstrates various ways to use LoRA in FedCore")
    
    # Example 1: EfficientNet
    try:
        example_efficientnet_lora()
    except Exception as e:
        print(f"[ERROR] Example 1 failed: {e}")
    
    # Example 2: Transformer (may skip if no transformers)
    try:
        example_transformer_lora()
    except Exception as e:
        print(f"[ERROR] Example 2 failed: {e}")
    
    # Example 3: Custom ResNet
    try:
        example_custom_model_lora()
    except Exception as e:
        print(f"[ERROR] Example 3 failed: {e}")
    
    # Example 4: Rank comparison
    try:
        example_lora_comparison()
    except Exception as e:
        print(f"[ERROR] Example 4 failed: {e}")
    
    # Example 5: Save/Load
    try:
        example_lora_save_load()
    except Exception as e:
        print(f"[ERROR] Example 5 failed: {e}")
    
    # Example 6: Multiple architectures
    try:
        example_lora_multiple_models()
    except Exception as e:
        print(f"[ERROR] Example 6 failed: {e}")
    
    # Example 7: API usage (conceptual)
    try:
        example_lora_with_api()
    except Exception as e:
        print(f"[ERROR] Example 7 failed: {e}")
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED")
    print("="*80)
    print("\nStage 10: Examples - COMPLETED")


if __name__ == "__main__":
    main()

