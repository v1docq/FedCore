"""
LLM Trainer implementation using transformers library
Real integration with transformers.Trainer
"""

from typing import Any, Dict, Optional, Iterable, Union
from enum import Enum

# Transformers imports
from torch.utils.data import Dataset
from transformers import (
    Trainer, 
    TrainingArguments, 
)
from transformers.trainer_callback import EarlyStoppingCallback
from datasets import Dataset
import numpy as np
from fedot.core.data.data import InputData

from fedcore.data.data import CompressionInputData
from fedcore.models.network_impl.utils._base import BaseTrainer
from fedcore.models.network_impl.utils.hooks_impl import FedCoreTransformersTrainer



class LLMTrainer(BaseTrainer):
    """
    LLM Trainer that implements our interfaces with real transformers.Trainer integration
    """
    
    def __init__(self, model, training_args: Optional[Dict] = None, **kwargs):
        super.__init__(self, params=training_args)
        
        self.model = model
        self._hooks = []
        self._additional_hooks = []
        
        self.default_training_args = {
            'output_dir': './llm_output',
            'num_train_epochs': 3,
            'per_device_train_batch_size': 4,
            'per_device_eval_batch_size': 4,
            'warmup_steps': 0,
            'lr_scheduler_type': 'linear',
            'weight_decay': 0.01,
            'logging_dir': './logs',
            'logging_steps': 10,
            'save_steps': 1000,
            'eval_steps': 1000,
            'evaluation_strategy': 'steps',
            'save_strategy': 'steps',
            'load_best_model_at_end': True,
            'metric_for_best_model': 'eval_loss',
            'greater_is_better': False,
        }
        
        if training_args:
            self.default_training_args.update(training_args)
        
        self._trainer = None
        self._training_args = None
        self._data_collator = None
        self._fedcore_callback = None

        self.trainer_objects = {
            'optimizer': None,
            'scheduler': None,
            'trainer': None
        }
        
    def register_additional_hooks(self, hooks: Iterable[Enum]) -> None:
        """Register additional hooks for training"""
        self._hooks.extend(hooks)
        
    def _init_hooks(self) -> None:
        """Initialize hooks for the model"""
        print("Initializing LLM hooks...")        
        
        self._fedcore_callback = FedCoreTransformersTrainer(
            hooks_params=self.default_training_args,
            model=self.model
        )
        
    def _prepare_data(self, input_data: Union[InputData, CompressionInputData]) -> Dict[str, Dataset]:
        """Convert InputData/CompressionInputData to transformers Dataset format"""
        train_data = self._dataloader_to_dataset(input_data.features.train_dataloader)
        eval_data = self._dataloader_to_dataset(input_data.features.val_dataloader)
            
        return {
            'train_dataset': train_data,
            'eval_dataset': eval_data
        }
            
    def _dataloader_to_dataset(self, dataloader) -> Dataset:
        """Convert PyTorch DataLoader to transformers Dataset"""
        data = []
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs, labels = batch[0], batch[1]
                inputs = inputs.cpu().numpy()
                labels = labels.cpu().numpy()
                    
                data.append({
                    'input_ids': inputs,
                    'labels': labels
                })
        return Dataset.from_list(data)

    def _create_trainer(self, datasets: Dict[str, Dataset]):
        """Create transformers trainer with FedCore integration"""
        allowed_training_args = {
            'output_dir', 'num_train_epochs', 'per_device_train_batch_size',
            'per_device_eval_batch_size', 'warmup_steps', 'lr_scheduler_type',
            'weight_decay', 'logging_dir', 'logging_steps', 'save_steps',
            'eval_steps', 'evaluation_strategy', 'save_strategy',
            'load_best_model_at_end', 'metric_for_best_model', 'greater_is_better'
        }
        
        filtered_args = {k: v for k, v in self.default_training_args.items() 
                        if k in allowed_training_args}
        
        self._training_args = TrainingArguments(**filtered_args)
        
        # Create callbacks list
        callbacks = []
        if self._fedcore_callback:
            callbacks.append(self._fedcore_callback)
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
        
        self._trainer = Trainer(
            model=self.model,
            args=self._training_args,
            train_dataset=datasets.get('train_dataset'),
            eval_dataset=datasets.get('eval_dataset'),
            data_collator=self._data_collator,
            callbacks=callbacks
        )
        
        # Connect trainer objects for hooks
        if self._fedcore_callback:
            self._fedcore_callback.trainer_objects['trainer'] = self._trainer

        self.trainer_objects['trainer'] = self._trainer

    def fit(self, input_data: Union[InputData, CompressionInputData], 
            supplementary_data: Optional[Dict] = None, loader_type: str = 'train') -> Any:
        """Train the model using InputData/CompressionInputData"""
        print(f"Training LLM model with {loader_type} data...")
        
        if not hasattr(self, '_fedcore_callback'):
            self._init_hooks()
        
        datasets = self._prepare_data(input_data)
        self._create_trainer(datasets)
        self.execute_hooks('start', epoch=0)
        
        train_result = self._trainer.train()
        print(f"Training completed. Loss: {getattr(train_result, 'training_loss', None)}")
        if self._fedcore_callback:
            self.history.update(self._fedcore_callback.history)
        
        return self.model
        
        
    def predict(self, input_data: Union[InputData, CompressionInputData], 
                output_mode: str = "default") -> Any:
        """Make predictions using InputData/CompressionInputData"""
        print(f"Making LLM predictions with {output_mode} mode...")
        
        if self._trainer is None:
            datasets = self._prepare_data(input_data)
            self._create_trainer(datasets)
            
        eval_dataset = self._dataloader_to_dataset(input_data.features.val_dataloader)
        predictions = self._trainer.predict(eval_dataset)
        
        return predictions
        
    def predict_for_fit(self, input_data: Union[InputData, CompressionInputData], 
                       output_mode: str = "default") -> Any:
        """Make predictions during training"""
        return self.predict(input_data, output_mode)
        
    def save_model(self, path: str) -> None:
        """Save the model using transformers approach"""
        print(f"Saving LLM model to {path}...")
        
        if self._trainer is not None:
            self._trainer.save_model(path)
        else:
            super().save_model(path)
        
    def load_model(self, path: str) -> None:
        """Load the model using transformers approach"""
        print(f"Loading LLM model from {path}...")

        super().load_model(path)
    
    @property
    def is_quantised(self) -> bool:
        """Check if model is quantized"""
        return getattr(self.model, '_is_quantised', False)
    
    @property
    def optimizer(self) -> Any:
        """Get optimizer from transformers trainer"""
        if self._fedcore_callback and 'optimizer' in self._fedcore_callback.trainer_objects:
            return self._fedcore_callback.trainer_objects['optimizer']
        return None
    
    @property
    def scheduler(self) -> Any:
        """Get scheduler from transformers trainer"""
        if self._fedcore_callback and 'scheduler' in self._fedcore_callback.trainer_objects:
            return self._fedcore_callback.trainer_objects['scheduler']
        return None
    
    @property
    def history(self) -> Dict:
        """Get training history from FedCore callback"""
        if self._fedcore_callback:
            return self._fedcore_callback.history
        return {}