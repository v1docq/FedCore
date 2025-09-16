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
    
    def __init__(self, model=None, params: Optional[Dict] = None, **kwargs):
        def _find_in_params(p: Optional[Dict], keys: Iterable[str]) -> Optional[Any]:
            if not isinstance(p, dict):
                return None
            for k in keys:
                if k in p and p[k] is not None:
                    return p[k]
            for v in p.values():
                if isinstance(v, dict):
                    found = _find_in_params(v, keys)
                    if found is not None:
                        return found
            return None
        if model is None and params and 'model' in params:
            model = params['model']
        
        if model is None and params and 'initial_assumption' in params:
            model = params['initial_assumption']

        if model is None and params and isinstance(params.get('custom_learning_params'), dict):
            nested = params.get('custom_learning_params')
            model = nested.get('model', model)

        if model is None and params:
            model = _find_in_params(params, keys=('model', 'initial_assumption', 'original_model'))
        
        super().__init__(model=model, params=params)
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
        
        if params:
            training_params = {k: v for k, v in params.items() 
                             if k not in ['model', 'tokenizer', 'custom_learning_params']}
            self.default_training_args.update(training_params)
        
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
        train_data = self._dataloader_to_dataset(input_data.train_dataloader)
        eval_data = self._dataloader_to_dataset(input_data.val_dataloader)
            
        return {
            'train_dataset': train_data,
            'eval_dataset': eval_data
        }
    def _prepare_text_generation_data(self, dataloader) -> Dataset:
        """Prepare data specifically for text generation"""
        data = []
        for batch in dataloader:
            if isinstance(batch, dict):
                input_ids = batch.get('input_ids', batch.get('inputs'))
                attention_mask = batch.get('attention_mask')
                labels = batch.get('labels', batch.get('targets'))

                if input_ids is None:
                    continue

                batch_size = input_ids.shape[0]
                for i in range(batch_size):
                    data.append({
                        'input_ids': input_ids[i].cpu().tolist(),
                        'attention_mask': None if attention_mask is None else attention_mask[i].cpu().tolist(),
                        'labels': None if labels is None else labels[i].cpu().tolist(),
                    })
        return Dataset.from_list(data)
            
    def _dataloader_to_dataset(self, dataloader) -> Dataset:
        data = []
        for batch in dataloader:
            if isinstance(batch, dict):
                input_ids = batch.get('input_ids')
                labels = batch.get('labels') if 'labels' in batch else batch.get('targets')
                attention_mask = batch.get('attention_mask')

                if input_ids is None:
                    continue

                batch_size = input_ids.shape[0]
                for i in range(batch_size):
                    data.append({
                        'input_ids': input_ids[i].cpu().tolist(),
                        'labels': None if labels is None else labels[i].cpu().tolist(),
                        'attention_mask': None if attention_mask is None else attention_mask[i].cpu().tolist(),
                    })
            elif isinstance(batch, (list, tuple)):
                # Expecting (inputs, labels, attention_mask) style tuples
                inputs = batch[0] if len(batch) >= 1 else None
                labels = batch[1] if len(batch) >= 2 else None
                attention_mask = batch[2] if len(batch) >= 3 else None

                if inputs is None:
                    continue

                batch_size = inputs.shape[0]
                for i in range(batch_size):
                    data.append({
                        'input_ids': inputs[i].cpu().tolist(),
                        'labels': None if labels is None else labels[i].cpu().tolist(),
                        'attention_mask': None if attention_mask is None else attention_mask[i].cpu().tolist(),
                    })
        return Dataset.from_list(data)

    def _create_trainer(self, datasets: Dict[str, Dataset]):
        """Create transformers trainer with FedCore integration"""
        # if self.model is None:
        #     raise ValueError("Model is None. Cannot create Trainer without a model.")
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
        
        if self.model is None:
            raise ValueError("LLMTrainer initialization failed: model is None after parameter resolution. Ensure 'model' or 'initial_assumption' is provided in config (including inside 'custom_learning_params').")

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
        
        # Final fallback: pull model from input_data if still missing
        if self.model is None:
            candidate_model = getattr(input_data, 'target', None)
            if candidate_model is None and hasattr(input_data, 'features'):
                candidate_model = getattr(input_data.features, 'target', None)
            if candidate_model is not None:
                self.model = candidate_model

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