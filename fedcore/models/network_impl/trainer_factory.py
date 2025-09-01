"""
Trainer Factory for creating appropriate trainers based on task type
"""

from typing import Any, Dict, Optional
from fedot.core.operations.operation_parameters import OperationParameters

from fedcore.models.network_impl.base_nn_model import BaseNeuralModel, BaseNeuralForecaster
from fedcore.models.network_impl.llm_trainer import LLMTrainer
from fedcore.models.network_impl.interfaces import ITrainer

def _analyze_model_architecture(model: Any) -> str:
    """
    Analyze model architecture to determine appropriate trainer type.
    
    Args:
        model: The model to analyze
        
    Returns:
        str: Trainer type ('llm', 'forecasting', 'general')
    """
    if model is None:
        return 'general'
    
    model_class_name = model.__class__.__name__.lower()
    
    llm_patterns = [
        'bert', 'gpt', 'transformer', 't5', 'roberta', 'distilbert',
        'albert', 'xlnet', 'electra', 'bart', 'llama', 'mistral',
        'bloom', 'opt', 'falcon', 'chatglm', 'qwen'
    ]
    
    for pattern in llm_patterns:
        if pattern in model_class_name:
            return 'llm'
    
    if hasattr(model, 'config'):
        config = model.config
        config_class_name = config.__class__.__name__.lower()
        
        transformer_config_patterns = [
            'bert', 'gpt', 'transformer', 't5', 'roberta', 'distilbert',
            'albert', 'xlnet', 'electra', 'bart', 'llama'
        ]
        
        for pattern in transformer_config_patterns:
            if pattern in config_class_name:
                return 'llm'
        
        transformer_attrs = [
            'num_hidden_layers', 'num_attention_heads', 'hidden_size',
            'vocab_size', 'max_position_embeddings', 'type_vocab_size'
        ]
        
        if any(hasattr(config, attr) for attr in transformer_attrs):
            return 'llm'
    
    forecasting_patterns = [
        'lstm', 'gru', 'rnn', 'tcn', 'temporal', 'time', 'forecast',
        'arima', 'prophet', 'ets', 'nbeats', 'nhits', 'transformerforecast'
    ]
    
    for pattern in forecasting_patterns:
        if pattern in model_class_name:
            return 'forecasting'
    
    if hasattr(model, 'modules'):
        for module in model.modules():
            module_name = module.__class__.__name__.lower()
            if any(pattern in module_name for pattern in ['lstm', 'gru', 'rnn', 'tcn']):
                return 'forecasting'
    
    return 'general'


def _get_trainer_class(model: Any, task_type: str, params: Dict) -> Type[ITrainer]:
    """
    Determine the appropriate trainer class based on model architecture and task type.
    
    Args:
        model: The model to train
        task_type: Task type from input data
        params: Training parameters
        
    Returns:
        Type[ITrainer]: Appropriate trainer class
    """
    architecture_type = _analyze_model_architecture(model)
    
    if architecture_type == 'llm':
        print(f"Creating LLMTrainer based on transformer architecture")
        return LLMTrainer
    
    elif architecture_type == 'forecasting':
        print(f"Creating BaseNeuralForecaster based on forecasting architecture")
        return BaseNeuralForecaster
    
    elif 'forecasting' in task_type.lower():
        print(f"Creating BaseNeuralForecaster based on task type: {task_type}")
        return BaseNeuralForecaster
    
    elif 'llm' in task_type.lower() or 'transformer' in task_type.lower():
        print(f"Creating LLMTrainer based on task type: {task_type}")
        return LLMTrainer
    
    else:
        print(f"Creating BaseNeuralModel for general task: {task_type}")
        return BaseNeuralModel


def create_trainer(
    task_type: str,
    params: Optional[OperationParameters] = None,
    model: Any = None,
    **kwargs
) -> ITrainer:
    """
    Create appropriate trainer based on model architecture and task type
    
    Args:
        task_type: Type of task ('forecasting', 'llm', 'classification', 'regression', etc.)
        params: Training parameters
        model: Model to train
        **kwargs: Additional arguments
        
    Returns:
        ITrainer: Appropriate trainer instance
    """
    
    # Convert params to dict if needed
    if params is not None and hasattr(params, 'to_dict'):
        params_dict = params.to_dict()
    else:
        params_dict = params or {}
    
    trainer_class = _get_trainer_class(model, task_type, params_dict)
    
    if trainer_class == LLMTrainer:
        return LLMTrainer(model=model, training_args=params_dict, **kwargs)
    else:
        return trainer_class(params_dict, **kwargs)


def create_trainer_from_input_data(
    input_data: Any,
    params: Optional[OperationParameters] = None,
    model: Any = None,
    **kwargs
) -> ITrainer:
    """
    Create appropriate trainer based on input data and model architecture
    
    Args:
        input_data: Input data with task information
        params: Training parameters
        model: Model to train
        **kwargs: Additional arguments
        
    Returns:
        ITrainer: Appropriate trainer instance
    """
    
    # Extract task type from input_data
    if hasattr(input_data, 'task') and hasattr(input_data.task, 'task_type'):
        task_type = input_data.task.task_type.value
    else:
        task_type = 'classification'  # default fallback
    
    if model is None and hasattr(input_data, 'target'):
        model = input_data.target
    
    return create_trainer(task_type, params, model, **kwargs)