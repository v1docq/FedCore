import torch
import torch.nn as nn
import shutil
import pytest

from enum import Enum

from fedcore.models.network_impl.llm_trainer import LLMTrainer


class DummyModel(nn.Module):
    def __init__(self, input_size=10, output_size=5):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, input_ids, labels=None):
        outputs = self.linear(input_ids.float())
        if labels is not None:
            loss = nn.CrossEntropyLoss()(outputs, labels)
            return {'loss': loss, 'logits': outputs}
        return {'logits': outputs}


class SimpleModel(nn.Module):
    def __init__(self, input_size=10, output_size=3):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, input_ids, labels=None):
        outputs = self.linear(input_ids.float())
        if labels is not None:
            loss = nn.CrossEntropyLoss()(outputs, labels)
            return {'loss': loss, 'logits': outputs}
        return {'logits': outputs}


class DummyInputData:
    def __init__(self, train_samples=4, val_samples=2, input_size=10, num_classes=3):
        self.features = self.DummyFeatures(train_samples, val_samples, input_size, num_classes)
    
    class DummyFeatures:
        def __init__(self, train_samples, val_samples, input_size, num_classes):
            train_data = [(torch.randn(train_samples, input_size), 
                          torch.randint(0, num_classes, (train_samples,)))]
            val_data = [(torch.randn(val_samples, input_size), 
                        torch.randint(0, num_classes, (val_samples,)))]
            
            self.train_dataloader = train_data
            self.val_dataloader = val_data


class TestHooks(Enum):
    TEST_HOOK = "test_hook"


@pytest.fixture
def dummy_model():
    return DummyModel()


@pytest.fixture
def simple_model():
    return SimpleModel()


@pytest.fixture
def dummy_input_data():
    return DummyInputData()


@pytest.fixture
def llm_trainer(dummy_model):
    return LLMTrainer(dummy_model)


@pytest.fixture
def custom_llm_trainer(dummy_model):
    custom_args = {
        'num_train_epochs': 1,
        'per_device_train_batch_size': 2,
        'output_dir': './test_output'
    }
    return LLMTrainer(dummy_model, training_args=custom_args)


def test_llm_trainer_creation(llm_trainer, custom_llm_trainer):
    """Test LLMTrainer creation with transformers integration"""
    assert llm_trainer is not None
    assert custom_llm_trainer is not None
    
    required_methods = [
        'fit', 'predict', 'save_model', 'load_model', 
        'register_additional_hooks', '_init_hooks'
    ]
    
    for method in required_methods:
        assert hasattr(llm_trainer, method), f"LLMTrainer should have {method} method"
        assert hasattr(custom_llm_trainer, method), f"Custom LLMTrainer should have {method} method"


def test_llm_trainer_hooks_initialization(llm_trainer):
    """Test LLMTrainer hooks initialization"""
    llm_trainer._init_hooks()
    assert hasattr(llm_trainer, '_callbacks') or hasattr(llm_trainer, '_hooks')


def test_llm_trainer_data_preparation(llm_trainer, dummy_input_data):
    """Test LLMTrainer data preparation"""
    datasets = llm_trainer._prepare_data(dummy_input_data)
    assert isinstance(datasets, dict)
    assert 'train_dataset' in datasets or 'eval_dataset' in datasets


def test_llm_trainer_transformers_trainer_creation(llm_trainer, dummy_input_data):
    """Test LLMTrainer transformers trainer creation"""
    llm_trainer._init_hooks()
    datasets = llm_trainer._prepare_data(dummy_input_data)
    llm_trainer._create_transformers_trainer(datasets)
    assert hasattr(llm_trainer, '_transformers_trainer')


def test_llm_trainer_is_quantized_property(llm_trainer):
    """Test LLMTrainer is_quantized property"""
    assert hasattr(llm_trainer, 'is_quantised')
    assert isinstance(llm_trainer.is_quantised, bool)


def test_llm_trainer_register_additional_hooks(llm_trainer):
    """Test LLMTrainer hook registration"""
    llm_trainer.register_additional_hooks([TestHooks])


def test_llm_trainer_save_load(simple_model, tmp_path):
    """Test LLMTrainer save and load methods"""
    trainer = LLMTrainer(simple_model)
    save_path = tmp_path / "test_model_save"
    
    trainer.save_model(str(save_path))
    assert save_path.exists()
    
    new_trainer = LLMTrainer(SimpleModel())
    new_trainer.load_model(str(save_path))


def test_llm_trainer_fit_predict(simple_model, dummy_input_data, tmp_path):
    """Test LLMTrainer fit and predict methods"""
    trainer = LLMTrainer(simple_model, training_args={
        'num_train_epochs': 1,
        'per_device_train_batch_size': 2,
        'output_dir': str(tmp_path / 'test_fit_output')
    })
    
    trained_model = trainer.fit(dummy_input_data)
    assert trained_model is not None
    
    predictions = trainer.predict(dummy_input_data)
    assert predictions is not None


@pytest.mark.skipif(not hasattr(LLMTrainer, '_create_transformers_callbacks'), 
                   reason="Method not available in this version")
def test_llm_trainer_callbacks_creation(llm_trainer):
    """Test LLMTrainer callbacks creation"""
    llm_trainer._init_hooks()
    llm_trainer._create_transformers_callbacks()
    assert hasattr(llm_trainer, '_callbacks')


def test_llm_trainer_with_minimal_args(dummy_model):
    """Test LLMTrainer with minimal arguments"""
    trainer = LLMTrainer(dummy_model, training_args={'num_train_epochs': 1})
    assert trainer is not None


def test_llm_trainer_properties(llm_trainer):
    """Test LLMTrainer properties"""
    assert hasattr(llm_trainer, 'optimizer')
    assert hasattr(llm_trainer, 'scheduler')
    assert hasattr(llm_trainer, 'model')


@pytest.fixture(autouse=True)
def cleanup_after_tests(tmp_path):
    """Cleanup temporary files after each test"""
    yield
    test_dirs = [
        './test_output',
        './test_fit_output', 
        './test_model_save',
        str(tmp_path / 'test_output'),
        str(tmp_path / 'test_fit_output'),
        str(tmp_path / 'test_model_save')
    ]
    
    for dir_path in test_dirs:
        shutil.rmtree(dir_path, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])