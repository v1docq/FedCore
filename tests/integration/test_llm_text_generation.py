"""
Test script for LLM text generation functionality in LLMTrainer.

Tests:
1. generate_long_text method
2. predict with output_mode="texts"
3. Integration with NLP metrics

Optimized for RTX 3060 (12GB VRAM) and 16GB RAM.
"""

import torch
import logging
from typing import List
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer

from fedcore.models.network_impl.llm_trainer import LLMTrainer
from fedcore.data.data import CompressionInputData
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.dataset_types import DataTypesEnum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use a small model that fits in RTX 3060 memory
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SimpleTextDataset(Dataset):
    """Simple dataset for text generation testing."""
    
    def __init__(self, texts: List[str], tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer
        self.references = texts.copy()  # For metrics
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=32,
            truncation=True
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
        }


def load_model_safely(model_name: str, device: str):
    """Load model with memory optimization."""
    logger.info(f"Loading model: {model_name}")
    logger.info(f"Device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Tokenizer loaded")
    
    # Load model with memory optimization
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        model = model.to(device)
    
    logger.info("Model loaded successfully")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer


def create_test_data(tokenizer, num_samples: int = 4):
    """Create test dataset with simple prompts."""
    test_prompts = [
        "The capital of France is",
        "In the future, AI will",
        "Machine learning is",
        "Python is a programming language",
    ][:num_samples]
    
    dataset = SimpleTextDataset(test_prompts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    return dataset, dataloader


def test_generate_long_text(trainer, input_data):
    """Test generate_long_text method."""
    logger.info("=" * 60)
    logger.info("Test 1: generate_long_text method")
    logger.info("=" * 60)
    
    try:
        generated_texts = trainer.generate_long_text(
            input_data=input_data,
            max_new_tokens=20,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
        
        logger.info(f"Generated {len(generated_texts)} texts")
        for i, text in enumerate(generated_texts):
            logger.info(f"  {i+1}. {text[:100]}...")
        
        assert len(generated_texts) > 0, "No texts generated"
        assert all(isinstance(t, str) for t in generated_texts), "Generated texts should be strings"
        
        logger.info("Test 1 passed: generate_long_text works correctly")
        return True
    except Exception as e:
        logger.error(f"Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_predict_with_texts_mode(trainer, input_data):
    """Test predict method with output_mode='texts'."""
    logger.info("=" * 60)
    logger.info("Test 2: predict with output_mode='texts'")
    logger.info("=" * 60)
    
    try:
        result = trainer.predict(
            input_data=input_data,
            output_mode="texts",
            max_new_tokens=15,
            temperature=0.8,
            do_sample=True
        )
        
        logger.info(f"Result type: {type(result)}")
        logger.info(f"Result.predict type: {type(result.predict)}")
        
        if isinstance(result.predict, list):
            logger.info(f"Generated {len(result.predict)} texts")
            for i, text in enumerate(result.predict):
                logger.info(f"  {i+1}. {text[:80]}...")
        else:
            logger.info(f"Result.predict value: {result.predict}")
        
        assert hasattr(result, 'predict'), "Result should have 'predict' attribute"
        assert isinstance(result.predict, list), "predict should be a list of strings"
        assert len(result.predict) > 0, "Should generate at least one text"
        assert all(isinstance(t, str) for t in result.predict), "All predictions should be strings"
        
        logger.info("Test 2 passed: predict with output_mode='texts' works correctly")
        return True
    except Exception as e:
        logger.error(f"Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics_integration(trainer, input_data):
    """Test NLP metrics integration."""
    logger.info("=" * 60)
    logger.info("Test 3: NLP metrics integration")
    logger.info("=" * 60)
    
    try:
        # Create a simple mock pipeline for metrics
        class MockPipeline:
            def __init__(self, trainer):
                self.trainer = trainer
            
            def predict(self, reference_data, output_mode, **kwargs):
                return self.trainer.predict(reference_data, output_mode=output_mode, **kwargs)
        
        pipeline = MockPipeline(trainer)
        
        # Test that predict returns texts correctly (which metrics will use)
        logger.info("Testing that predict returns texts correctly...")
        
        result = pipeline.predict(input_data, output_mode="texts", max_new_tokens=10)
        assert isinstance(result.predict, list), "Should return list of strings"
        assert all(isinstance(t, str) for t in result.predict), "All predictions should be strings"
        
        logger.info("Test 3 passed: predict returns texts correctly for metrics")
        logger.info("   (Full metric integration test requires evaluate package setup)")
        return True
        
    except Exception as e:
        logger.error(f"Test 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation_parameters(trainer, input_data):
    """Test different generation parameters."""
    logger.info("=" * 60)
    logger.info("Test 4: Generation parameters")
    logger.info("=" * 60)
    
    try:
        # Test greedy decoding
        logger.info("Testing greedy decoding (do_sample=False)...")
        result1 = trainer.predict(
            input_data=input_data,
            output_mode="texts",
            max_new_tokens=10,
            do_sample=False,
            temperature=None
        )
        assert len(result1.predict) > 0
        logger.info("  Greedy decoding works")
        
        # Test sampling
        logger.info("Testing sampling (do_sample=True)...")
        result2 = trainer.predict(
            input_data=input_data,
            output_mode="texts",
            max_new_tokens=10,
            do_sample=True,
            temperature=0.7
        )
        assert len(result2.predict) > 0
        logger.info("  Sampling works")
        
        # Test with different max_new_tokens
        logger.info("Testing different max_new_tokens...")
        result3 = trainer.predict(
            input_data=input_data,
            output_mode="texts",
            max_new_tokens=5,
            do_sample=False,
            temperature=None
        )
        assert len(result3.predict) > 0
        logger.info("  Different max_new_tokens works")
        
        logger.info("Test 4 passed: Generation parameters work correctly")
        return True
    except Exception as e:
        logger.error(f"Test 4 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("LLM TEXT GENERATION TEST SUITE")
    logger.info("=" * 60)
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Device: {DEVICE}")
    logger.info("")
    
    try:
        # Load model
        model, tokenizer = load_model_safely(MODEL_NAME, DEVICE)
        
        # Create trainer
        trainer_params = {
            "model": model,
            "tokenizer": tokenizer,
            # "max_length": 50,  # Default max length
            "temperature": 0.7,  # Default temperature
        }
        trainer = LLMTrainer(params=trainer_params)
        logger.info("Trainer created")
        
        # Create test data
        test_dataset, test_dataloader = create_test_data(tokenizer, num_samples=4)
        
        input_data = CompressionInputData(
            val_dataloader=test_dataloader,
            task=Task(TaskTypesEnum.classification),
            data_type=DataTypesEnum.table,
        )
        # Store references for metrics
        input_data.features = type('obj', (object,), {
            'val_dataloader': test_dataloader,
            'task': Task(TaskTypesEnum.classification)
        })()
        
        logger.info(f"Test data created: {len(test_dataset)} samples")
        logger.info("")
        
        # Run tests
        results = []
        
        results.append(("generate_long_text", test_generate_long_text(trainer, input_data)))
        results.append(("predict with texts mode", test_predict_with_texts_mode(trainer, input_data)))
        results.append(("metrics integration", test_metrics_integration(trainer, input_data)))
        results.append(("generation parameters", test_generation_parameters(trainer, input_data)))
        
        # Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "PASSED" if result else "FAILED"
            logger.info(f"  {test_name}: {status}")
        
        logger.info("")
        logger.info(f"Total: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("All tests passed!")
            return 0
        else:
            logger.warning(f"{total - passed} test(s) failed")
            return 1
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleanup completed")


if __name__ == "__main__":
    exit(main())
