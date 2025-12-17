import pytest
from unittest.mock import Mock, MagicMock
from fedot.core.operations.operation_parameters import OperationParameters


class TestTrainerFactory:

    def test_imports(self):
        from fedcore.models.network_impl.interfaces import ITrainer
        from fedcore.models.network_impl.trainer_factory import (
            create_trainer, 
            create_trainer_from_input_data,
            _analyze_model_architecture,
            _get_trainer_class
        )
        from fedcore.models.network_impl.base_nn_model import BaseNeuralModel, BaseNeuralForecaster
        from fedcore.models.network_impl.llm_trainer import LLMTrainer
        
        assert True

    def test_analyze_model_architecture_none(self):
        from fedcore.models.network_impl.trainer_factory import _analyze_model_architecture
        
        result = _analyze_model_architecture(None)
        assert result == 'general'

    def test_analyze_model_architecture_llm_by_name(self):
        from fedcore.models.network_impl.trainer_factory import _analyze_model_architecture
        
        class MockBertModel:
            def __init__(self):
                self.__class__.__name__ = "BertForSequenceClassification"
        
        model = MockBertModel()
        result = _analyze_model_architecture(model)
        assert result == 'llm'

    def test_analyze_model_architecture_llm_by_config(self):
        from fedcore.models.network_impl.trainer_factory import _analyze_model_architecture
        
        class MockConfig:
            def __init__(self):
                self.num_hidden_layers = 12
                self.num_attention_heads = 12
                self.hidden_size = 768
        
        class MockModelWithConfig:
            def __init__(self):
                self.__class__.__name__ = "CustomModel"
                self.config = MockConfig()
        
        model = MockModelWithConfig()
        result = _analyze_model_architecture(model)
        assert result == 'llm'

    def test_analyze_model_architecture_forecasting(self):
        from fedcore.models.network_impl.trainer_factory import _analyze_model_architecture
        
        class MockLSTMModel:
            def __init__(self):
                self.__class__.__name__ = "LSTMForecaster"
        
        model = MockLSTMModel()
        result = _analyze_model_architecture(model)
        assert result == 'forecasting'

    def test_analyze_model_architecture_general(self):
        from fedcore.models.network_impl.trainer_factory import _analyze_model_architecture
        
        class MockGeneralModel:
            def __init__(self):
                self.__class__.__name__ = "LinearClassifier"
        
        model = MockGeneralModel()
        result = _analyze_model_architecture(model)
        assert result == 'general'

    def test_get_trainer_class_llm_architecture(self):
        from fedcore.models.network_impl.trainer_factory import _get_trainer_class, LLMTrainer
        
        # Mock LLM model
        class MockBertModel:
            def __init__(self):
                self.__class__.__name__ = "BertModel"
        
        model = MockBertModel()
        trainer_class = _get_trainer_class(model, "classification", {})
        assert trainer_class == LLMTrainer

    def test_get_trainer_class_forecasting_architecture(self):
        from fedcore.models.network_impl.trainer_factory import _get_trainer_class, BaseNeuralForecaster
        
        class MockLSTMModel:
            def __init__(self):
                self.__class__.__name__ = "LSTMModel"
        
        model = MockLSTMModel()
        trainer_class = _get_trainer_class(model, "regression", {})
        assert trainer_class == BaseNeuralForecaster

    def test_get_trainer_class_forecasting_task(self):
        from fedcore.models.network_impl.trainer_factory import _get_trainer_class, BaseNeuralForecaster
        
        class MockGeneralModel:
            def __init__(self):
                self.__class__.__name__ = "LinearModel"
        
        model = MockGeneralModel()
        trainer_class = _get_trainer_class(model, "time_series_forecasting", {})
        assert trainer_class == BaseNeuralForecaster

    def test_get_trainer_class_llm_task(self):
        from fedcore.models.network_impl.trainer_factory import _get_trainer_class, LLMTrainer
        
        class MockGeneralModel:
            def __init__(self):
                self.__class__.__name__ = "LinearModel"
        
        model = MockGeneralModel()
        trainer_class = _get_trainer_class(model, "llm_text_generation", {})
        assert trainer_class == LLMTrainer

    def test_get_trainer_class_general(self):
        from fedcore.models.network_impl.trainer_factory import _get_trainer_class, BaseNeuralModel
        
        class MockGeneralModel:
            def __init__(self):
                self.__class__.__name__ = "LinearModel"
        
        model = MockGeneralModel()
        trainer_class = _get_trainer_class(model, "classification", {})
        assert trainer_class == BaseNeuralModel

    def test_create_trainer_with_params_object(self):
        from fedcore.models.network_impl.trainer_factory import create_trainer, BaseNeuralModel
        
        params = OperationParameters()
        params_dict = {'learning_rate': 0.01, 'batch_size': 32}
        params.to_dict = Mock(return_value=params_dict)
        
        trainer = create_trainer("classification", params=params)
        assert isinstance(trainer, BaseNeuralModel)

    def test_create_trainer_with_params_dict(self):
        from fedcore.models.network_impl.trainer_factory import create_trainer, BaseNeuralModel
        
        params_dict = {'learning_rate': 0.01, 'batch_size': 32}
        trainer = create_trainer("classification", params=params_dict)
        assert isinstance(trainer, BaseNeuralModel)

    def test_create_trainer_with_model(self):
        from fedcore.models.network_impl.trainer_factory import create_trainer, BaseNeuralModel
        
        class MockModel:
            def __init__(self):
                self.__class__.__name__ = "LinearModel"
        
        model = MockModel()
        trainer = create_trainer("classification", model=model)
        assert isinstance(trainer, BaseNeuralModel)

    def test_create_trainer_llm(self):
        from fedcore.models.network_impl.trainer_factory import create_trainer, LLMTrainer
        
        class MockBertModel:
            def __init__(self):
                self.__class__.__name__ = "BertModel"
        
        model = MockBertModel()
        trainer = create_trainer("classification", model=model)
        assert isinstance(trainer, LLMTrainer)

    def test_create_trainer_forecasting(self):
        from fedcore.models.network_impl.trainer_factory import create_trainer, BaseNeuralForecaster
        
        class MockLSTMModel:
            def __init__(self):
                self.__class__.__name__ = "LSTMModel"
        
        model = MockLSTMModel()
        trainer = create_trainer("regression", model=model)
        assert isinstance(trainer, BaseNeuralForecaster)

    def test_create_trainer_from_input_data_with_task(self):
        from fedcore.models.network_impl.trainer_factory import create_trainer_from_input_data, BaseNeuralModel
        
        mock_task = Mock()
        mock_task.task_type = Mock()
        mock_task.task_type.value = "classification"
        
        mock_input_data = Mock()
        mock_input_data.task = mock_task
        
        trainer = create_trainer_from_input_data(mock_input_data)
        assert isinstance(trainer, BaseNeuralModel)

    def test_create_trainer_from_input_data_with_target(self):
        from fedcore.models.network_impl.trainer_factory import create_trainer_from_input_data, BaseNeuralModel
        
        mock_target = Mock()
        mock_target.__class__.__name__ = "LinearModel"
        
        mock_input_data = Mock()
        mock_input_data.target = mock_target
        
        def mock_getattr(name):
            if name == 'task':
                raise AttributeError()
            return getattr(mock_input_data, name)
        
        mock_input_data.__getattr__ = mock_getattr
        
        trainer = create_trainer_from_input_data(mock_input_data)
        assert isinstance(trainer, BaseNeuralModel)

    def test_create_trainer_from_input_data_default(self):
        from fedcore.models.network_impl.trainer_factory import create_trainer_from_input_data, BaseNeuralModel
        
        mock_input_data = Mock()
        
        def mock_getattr(name):
            raise AttributeError()
        
        mock_input_data.__getattr__ = mock_getattr
        
        trainer = create_trainer_from_input_data(mock_input_data)
        assert isinstance(trainer, BaseNeuralModel)

    def test_trainer_implements_interfaces(self):
        from fedcore.models.network_impl.trainer_factory import create_trainer
        from fedcore.models.network_impl.interfaces import ITrainer
        
        trainers = [
            create_trainer("classification"),
            create_trainer("forecasting"),
        ]
        
        for trainer in trainers:
            assert isinstance(trainer, ITrainer)

    def test_trainer_methods_exist(self):
        """Test that trainers have required methods"""
        from fedcore.models.network_impl.trainer_factory import create_trainer
        
        trainers = [
            create_trainer("classification"),
            create_trainer("forecasting"),
        ]
        
        required_methods = ['train', 'evaluate', 'predict', 'get_model', 'save_model', 'load_model']
        
        for trainer in trainers:
            for method in required_methods:
                assert hasattr(trainer, method), f"Trainer {type(trainer).__name__} missing method {method}"


# Parameterized tests for better coverage
@pytest.mark.parametrize("model_name,expected_type", [
    ("BertModel", "llm"),
    ("GPT2Model", "llm"),
    ("TransformerModel", "llm"),
    ("LSTMModel", "forecasting"),
    ("GRUModel", "forecasting"),
    ("TCNModel", "forecasting"),
    ("LinearModel", "general"),
    ("LogisticRegression", "general"),
])
def test_analyze_model_architecture_parameterized(model_name, expected_type):
    """Parameterized test for model architecture analysis"""
    from fedcore.models.network_impl.trainer_factory import _analyze_model_architecture
    
    class MockModel:
        def __init__(self, name):
            self.__class__.__name__ = name
    
    model = MockModel(model_name)
    result = _analyze_model_architecture(model)
    assert result == expected_type


@pytest.mark.parametrize("task_type,model_name,expected_trainer", [
    ("classification", "LinearModel", "BaseNeuralModel"),
    ("forecasting", "LinearModel", "BaseNeuralForecaster"),
    ("time_series", "LinearModel", "BaseNeuralForecaster"),
    ("llm_generation", "LinearModel", "LLMTrainer"),
    ("classification", "BertModel", "LLMTrainer"),
    ("regression", "LSTMModel", "BaseNeuralForecaster"),
])
def test_get_trainer_class_parameterized(task_type, model_name, expected_trainer):
    """Parameterized test for getting trainer class"""
    from fedcore.models.network_impl.trainer_factory import _get_trainer_class
    from fedcore.models.network_impl.base_nn_model import BaseNeuralModel, BaseNeuralForecaster
    from fedcore.models.network_impl.llm_trainer import LLMTrainer
    
    class MockModel:
        def __init__(self, name):
            self.__class__.__name__ = name
    
    model = MockModel(model_name)
    trainer_class = _get_trainer_class(model, task_type, {})
    
    expected_classes = {
        "BaseNeuralModel": BaseNeuralModel,
        "BaseNeuralForecaster": BaseNeuralForecaster,
        "LLMTrainer": LLMTrainer
    }
    
    assert trainer_class == expected_classes[expected_trainer]