"""
Integration tests for LoRA implementation in FedCore.

Tests cover:
1. EfficientNet (Conv2d layers)
2. Transformers (HF + LLMTrainer)
3. ResNet (Custom model + BaseNeuralModel)
4. API integration
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from fedcore.algorithm.low_rank.lora_operation import BaseLoRA

try:
    import pytest
except ImportError:
    # Create dummy pytest for standalone running
    class pytest:
        @staticmethod
        def skip(reason):
            raise Exception(f"SKIP: {reason}")
        
        class mark:
            @staticmethod
            def skipif(condition, reason=""):
                def decorator(func):
                    return func
                return decorator


class TestLoraEfficientNet:
    """Test LoRA on EfficientNet (convolutional layers)."""
    
    def test_lora_conv2d_application(self):
        """Test LoRA application to Conv2d layers."""
        from torchvision.models import efficientnet_b0
        
        # Create EfficientNet model
        model = efficientnet_b0(pretrained=False)
        model.eval()
        
        # Apply LoRA
        lora = BaseLoRA(params={
            'lora_r': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.1,
            'lora_target_modules': ['features'],  # Target convolutional features
            'use_peft': False
        })
        
        lora_params = {
            'lora_r': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.1,
            'lora_target_modules': [],
            'use_peft': False
        }
        
        # Apply LoRA to model
        model_with_lora = lora._apply_lora_to_model(model, lora_params)
        
        # Check that some Conv2d layers have LoRA applied
        has_lora_conv = False
        for name, module in model_with_lora.named_modules():
            if 'lora' in name.lower():
                has_lora_conv = True
                break
        
        assert has_lora_conv, "LoRA should be applied to some Conv2d layers"
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = model_with_lora(x)
        assert output.shape == (2, 1000), f"Expected shape (2, 1000), got {output.shape}"
    
    def test_lora_conv2d_training(self):
        """Test that LoRA parameters are trainable on Conv2d."""
        from torchvision.models import efficientnet_b0
        
        model = efficientnet_b0(pretrained=False)
        
        lora = BaseLoRA(params={
            'lora_r': 4,
            'lora_alpha': 8,
            'lora_dropout': 0.0,
            'lora_target_modules': [],
            'use_peft': False
        })
        
        lora_params = {
            'lora_r': 4,
            'lora_alpha': 8,
            'lora_dropout': 0.0,
            'lora_target_modules': [],
            'use_peft': False
        }
        
        model_with_lora = lora._apply_lora_to_model(model, lora_params)
        lora._freeze_non_lora_parameters(model_with_lora)
        
        # Count trainable parameters
        trainable = sum(p.numel() for p in model_with_lora.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model_with_lora.parameters())
        
        # Should have some trainable parameters (LoRA)
        assert trainable > 0, "Should have trainable LoRA parameters"
        # Should be much less than total (parameter efficient)
        assert trainable < total * 0.5, "LoRA should be parameter efficient"
        
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


class TestLoraResNet:
    """Test LoRA on ResNet (custom model)."""
    
    def test_lora_resnet_linear(self):
        """Test LoRA application to ResNet Linear layers."""
        from torchvision.models import resnet18
        
        # Create ResNet18
        model = resnet18(pretrained=False)
        model.eval()
        
        # Apply LoRA
        lora = BaseLoRA(params={
            'lora_r': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.1,
            'lora_target_modules': ['fc'],  # Target fully connected layer
            'use_peft': False
        })
        
        lora_params = {
            'lora_r': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.1,
            'lora_target_modules': ['fc'],
            'use_peft': False
        }
        
        model_with_lora = lora._apply_lora_to_model(model, lora_params)
        
        # Check that fc layer has LoRA
        fc_has_lora = any('fc' in name and 'lora' in name.lower() 
                          for name, _ in model_with_lora.named_modules())
        
        assert fc_has_lora, "FC layer should have LoRA applied"
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = model_with_lora(x)
        assert output.shape == (2, 1000), f"Expected shape (2, 1000), got {output.shape}"
    
    def test_lora_resnet_parameter_efficiency(self):
        """Test parameter efficiency on ResNet."""
        from torchvision.models import resnet18
        
        model = resnet18(pretrained=False)
        
        lora = BaseLoRA(params={
            'lora_r': 4,
            'lora_alpha': 8,
            'lora_dropout': 0.0,
            'lora_target_modules': [],
            'use_peft': False
        })
        
        lora_params = {
            'lora_r': 4,
            'lora_alpha': 8,
            'lora_dropout': 0.0,
            'lora_target_modules': [],
            'use_peft': False
        }
        
        model_with_lora = lora._apply_lora_to_model(model, lora_params)
        lora._freeze_non_lora_parameters(model_with_lora)
        
        # Count parameters
        trainable = sum(p.numel() for p in model_with_lora.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model_with_lora.parameters())
        
        assert trainable > 0, "Should have trainable parameters"
        assert trainable < total * 0.3, "LoRA should train less than 30% of parameters"
        
        print(f"ResNet18 - Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


class TestLoraHuggingFace:
    """Test LoRA on HuggingFace models."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_lora_hf_detection(self):
        """Test HuggingFace model detection."""
        try:
            from transformers import AutoModel
            
            # Load small model
            model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
            
            lora = BaseLoRA(params={'lora_r': 4})
            
            # Check HF detection
            is_hf = lora._is_huggingface_model(model)
            assert is_hf, "Should detect HuggingFace model"
            
        except ImportError:
            pytest.skip("transformers not installed")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_lora_hf_with_peft(self):
        """Test LoRA with PEFT library."""
        try:
            from transformers import AutoModel
            from peft import get_peft_model
            
            # Load small model
            model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
            
            lora = BaseLoRA(params={
                'lora_r': 8,
                'lora_alpha': 16,
                'lora_target_modules': ['query', 'value'],
                'use_peft': True
            })
            
            lora_params = {
                'lora_r': 8,
                'lora_alpha': 16,
                'lora_target_modules': ['query', 'value'],
                'use_peft': True
            }
            
            # Apply LoRA
            model_with_lora = lora._apply_lora_huggingface(model, lora_params)
            
            # Check trainable parameters
            trainable = sum(p.numel() for p in model_with_lora.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model_with_lora.parameters())
            
            assert trainable > 0, "Should have trainable LoRA parameters"
            assert trainable < total * 0.2, "PEFT LoRA should be parameter efficient"
            
            print(f"PEFT LoRA - Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
            
        except ImportError as e:
            pytest.skip(f"Required library not installed: {e}")
    
    def test_lora_hf_fallback(self):
        """Test fallback to fedcore implementation when PEFT unavailable."""
        try:
            from transformers import AutoModel
            
            model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
            
            lora = BaseLoRA(params={
                'lora_r': 4,
                'lora_alpha': 8,
                'use_peft': False  # Force fedcore implementation
            })
            
            lora_params = {
                'lora_r': 4,
                'lora_alpha': 8,
                'use_peft': False
            }
            
            # Should use fedcore implementation
            model_with_lora = lora._apply_lora_huggingface(model, lora_params)
            
            assert model_with_lora is not None, "Should apply LoRA with fedcore implementation"
            
        except ImportError:
            pytest.skip("transformers not installed")


class TestLoraTargetModules:
    """Test LoRA target module selection."""
    
    def test_target_modules_selection(self):
        """Test that target_modules parameter works correctly."""
        from torchvision.models import resnet18
        
        model = resnet18(pretrained=False)
        
        # Target only fc layer
        lora = BaseLoRA(params={
            'lora_r': 4,
            'lora_alpha': 8,
            'lora_target_modules': ['fc'],
            'use_peft': False
        })
        
        lora_params = {
            'lora_r': 4,
            'lora_alpha': 8,
            'lora_target_modules': ['fc'],
            'use_peft': False
        }
        
        model_with_lora = lora._apply_lora_to_model(model, lora_params)
        
        # Check that only fc has LoRA
        lora_layers = [name for name, _ in model_with_lora.named_modules() if 'lora' in name.lower()]
        
        # All LoRA layers should be in fc
        assert all('fc' in name for name in lora_layers), "LoRA should only be in fc layer"
    
    def test_empty_target_modules(self):
        """Test that empty target_modules applies to all suitable layers."""
        from torchvision.models import resnet18
        
        model = resnet18(pretrained=False)
        
        # Empty target modules - should apply to all suitable layers
        lora = BaseLoRA(params={
            'lora_r': 4,
            'lora_alpha': 8,
            'lora_target_modules': [],
            'use_peft': False
        })
        
        lora_params = {
            'lora_r': 4,
            'lora_alpha': 8,
            'lora_target_modules': [],
            'use_peft': False
        }
        
        model_with_lora = lora._apply_lora_to_model(model, lora_params)
        lora._freeze_non_lora_parameters(model_with_lora)
        
        # Should have LoRA applied to multiple layers
        trainable = sum(p.numel() for p in model_with_lora.parameters() if p.requires_grad)
        assert trainable > 0, "Should have trainable LoRA parameters"


class TestLoraBackward:
    """Test LoRA backward pass and gradient flow."""
    
    def test_lora_gradients(self):
        """Test that gradients flow only through LoRA parameters."""
        from torchvision.models import resnet18
        
        model = resnet18(pretrained=False)
        
        lora = BaseLoRA(params={
            'lora_r': 4,
            'lora_alpha': 8,
            'lora_target_modules': ['fc'],
            'use_peft': False
        })
        
        lora_params = {
            'lora_r': 4,
            'lora_alpha': 8,
            'lora_target_modules': ['fc'],
            'use_peft': False
        }
        
        model_with_lora = lora._apply_lora_to_model(model, lora_params)
        lora._freeze_non_lora_parameters(model_with_lora)
        
        # Forward pass
        x = torch.randn(2, 3, 224, 224)
        y = torch.randn(2, 1000)
        output = model_with_lora(x)
        
        # Backward pass
        loss = nn.MSELoss()(output, y)
        loss.backward()
        
        # Check gradients
        lora_grads_exist = False
        base_grads_exist = False
        
        for name, param in model_with_lora.named_parameters():
            if param.grad is not None:
                if 'lora' in name.lower():
                    lora_grads_exist = True
                else:
                    base_grads_exist = True
        
        assert lora_grads_exist, "LoRA parameters should have gradients"
        assert not base_grads_exist, "Base parameters should NOT have gradients"


if __name__ == "__main__":
    print("Running LoRA Integration Tests")
    print("=" * 80)
    
    # Run tests
    test_efficientnet = TestLoraEfficientNet()
    test_resnet = TestLoraResNet()
    test_hf = TestLoraHuggingFace()
    test_targets = TestLoraTargetModules()
    test_backward = TestLoraBackward()
    
    print("\n[TEST 1] EfficientNet Conv2d Application")
    try:
        test_efficientnet.test_lora_conv2d_application()
        print("[PASS] Conv2d LoRA application works")
    except Exception as e:
        print(f"[FAIL] {e}")
    
    print("\n[TEST 2] EfficientNet Training")
    try:
        test_efficientnet.test_lora_conv2d_training()
        print("[PASS] Conv2d LoRA training works")
    except Exception as e:
        print(f"[FAIL] {e}")
    
    print("\n[TEST 3] ResNet Linear Layers")
    try:
        test_resnet.test_lora_resnet_linear()
        print("[PASS] ResNet LoRA application works")
    except Exception as e:
        print(f"[FAIL] {e}")
    
    print("\n[TEST 4] ResNet Parameter Efficiency")
    try:
        test_resnet.test_lora_resnet_parameter_efficiency()
        print("[PASS] ResNet LoRA parameter efficiency OK")
    except Exception as e:
        print(f"[FAIL] {e}")
    
    print("\n[TEST 5] HuggingFace Detection")
    try:
        test_hf.test_lora_hf_detection()
        print("[PASS] HF model detection works")
    except Exception as e:
        print(f"[SKIP/FAIL] {e}")
    
    print("\n[TEST 6] Target Modules Selection")
    try:
        test_targets.test_target_modules_selection()
        print("[PASS] Target modules selection works")
    except Exception as e:
        print(f"[FAIL] {e}")
    
    print("\n[TEST 7] Empty Target Modules")
    try:
        test_targets.test_empty_target_modules()
        print("[PASS] Empty target modules works")
    except Exception as e:
        print(f"[FAIL] {e}")
    
    print("\n[TEST 8] Gradient Flow")
    try:
        test_backward.test_lora_gradients()
        print("[PASS] Gradient flow correct")
    except Exception as e:
        print(f"[FAIL] {e}")
    
    print("\n" + "=" * 80)
    print("Integration tests completed")

