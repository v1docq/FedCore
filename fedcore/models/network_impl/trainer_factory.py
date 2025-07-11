"""
Trainer Factory for creating appropriate trainers based on task type
"""

from typing import Any, Dict, Optional
from fedot.core.operations.operation_parameters import OperationParameters

from fedcore.models.network_impl.base_nn_model import BaseNeuralModel, BaseNeuralForecaster
from fedcore.models.network_impl.llm_trainer import LLMTrainer
from fedcore.models.network_impl.interfaces import ITrainer


def create_trainer(
    task_type: str,
    params: Optional[OperationParameters] = None,
    model: Any = None,
    **kwargs
) -> ITrainer:
    """
    Create appropriate trainer based on task type
    
    Args:
        task_type: Type of task ('forecasting', 'llm', 'classification', 'regression', etc.)
        params: Training parameters
        model: Model to train (for LLM trainer)
        **kwargs: Additional arguments
        
    Returns:
        ITrainer: Appropriate trainer instance
    """
    
    # Convert params to dict if needed
    if params is not None and hasattr(params, 'to_dict'):
        params_dict = params.to_dict()
    else:
        params_dict = params or {}
    
    # Choose trainer based on task type
    if 'forecasting' in task_type.lower():
        print(f"Creating BaseNeuralForecaster for {task_type}")
        return BaseNeuralForecaster(params)
    
    elif 'llm' in task_type.lower() or 'transformer' in task_type.lower():
        print(f"Creating LLMTrainer for {task_type}")
        return LLMTrainer(model=model, **kwargs)
    
    else:
        print(f"Creating BaseNeuralModel for {task_type}")
        return BaseNeuralModel(params)


def create_trainer_from_input_data(
    input_data: Any,
    params: Optional[OperationParameters] = None,
    model: Any = None,
    **kwargs
) -> ITrainer:
    """
    Create appropriate trainer based on input data task type
    
    Args:
        input_data: Input data with task information
        params: Training parameters
        model: Model to train (for LLM trainer)
        **kwargs: Additional arguments
        
    Returns:
        ITrainer: Appropriate trainer instance
    """
    
    # Extract task type from input_data
    if hasattr(input_data, 'task') and hasattr(input_data.task, 'task_type'):
        task_type = input_data.task.task_type.value
    else:
        task_type = 'classification'  # default fallback
    
    return create_trainer(task_type, params, model, **kwargs) 