"""Compatibility import for the public trainer factories."""
from .utils.trainer_factory import create_trainer, create_trainer_from_input_data, _get_trainer_class, _analyze_model_architecture

get_trainer_class = _get_trainer_class

__all__ = ['create_trainer', 'create_trainer_from_input_data', 'get_trainer_class']
