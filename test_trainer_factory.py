#!/usr/bin/env python3
"""
Simple test for trainer factory
"""

def test_imports():
    """Test if all imports work correctly"""
    try:
        print("Testing imports...")
        
        # Test basic imports
        from fedcore.models.network_impl.interfaces import ITrainer, IHookable
        print("✅ Interfaces imported successfully")
        
        from fedcore.models.network_impl.trainer_factory import create_trainer, create_trainer_from_input_data
        print("✅ Trainer factory imported successfully")
        
        from fedcore.models.network_impl.base_nn_model import BaseNeuralModel, BaseNeuralForecaster
        print("✅ Base neural models imported successfully")
        
        from fedcore.models.network_impl.llm_trainer import LLMTrainer
        print("✅ LLM trainer imported successfully")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_factory():
    """Test factory functions"""
    try:
        print("\nTesting factory functions...")
        
        from fedcore.models.network_impl.trainer_factory import create_trainer
        
        # Test with different task types
        trainer1 = create_trainer("classification")
        print(f"✅ Created trainer for classification: {type(trainer1).__name__}")
        
        trainer2 = create_trainer("forecasting")
        print(f"✅ Created trainer for forecasting: {type(trainer2).__name__}")
        
        trainer3 = create_trainer("llm")
        print(f"✅ Created trainer for LLM: {type(trainer3).__name__}")
        
        print("\n🎉 Factory functions work correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Factory test error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Trainer Factory Implementation")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test factory
        factory_ok = test_factory()
        
        if factory_ok:
            print("\n🎉 All tests passed! Your implementation is working correctly.")
        else:
            print("\n❌ Factory tests failed.")
    else:
        print("\n❌ Import tests failed.") 