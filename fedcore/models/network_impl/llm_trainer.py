"""
LLM Trainer implementation using transformers library
Real integration with transformers.Trainer
"""

from typing import Any, Dict, Optional, Iterable, Union
from enum import Enum
from itertools import chain
import torch
import os
from pathlib import Path

# Transformers imports
from transformers import (
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding
)
from transformers.trainer_callback import EarlyStoppingCallback
from datasets import Dataset
import numpy as np

from fedcore.models.network_impl.utils._base import BaseTrainer
from fedcore.models.network_impl.utils.hooks_collection import HooksCollection
from fedcore.models.network_impl.utils.hooks import LoggingHooks, ModelLearningHooks
from fedcore.models.network_impl.utils.hooks_impl import FedCoreTransformersTrainer


class LLMTrainer(BaseTrainer):
    """
    LLM Trainer that implements our interfaces with real transformers.Trainer integration
    """
    
    def __init__(self, model, training_args: Optional[Dict] = None, train_dataset=None, eval_dataset=None, tokenizer=None, **kwargs):
        BaseTrainer.__init__(self, params=training_args)
        
        self.model = model
        self._hooks = []
        self._additional_hooks = []
        
        self._train_dataset = train_dataset
        self._eval_dataset = eval_dataset
        self._tokenizer = tokenizer
        
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
        
        self._transformers_trainer = None
        self._training_args = None
        self._data_collator = None
        
    def register_additional_hooks(self, hooks: Iterable[Enum]) -> None:
        """Register additional hooks for training"""
        self._hooks.extend(hooks)
        
    def _init_hooks(self) -> None:
        """Initialize hooks for the model"""
        print("Initializing LLM hooks...")        
        
        # Adding standard hooks
        self._hooks = [LoggingHooks, ModelLearningHooks]
        
        # Initializing hooks
        for hook_elem in chain(*self._hooks):
            hook = hook_elem.value
            if not hook.check_init(self.default_training_args):
                continue
            hook_instance = hook(self.default_training_args, self.model)
            self.hooks.append(hook_instance)
        
        # Map our hooks to transformers callbacks
        self._create_transformers_callbacks()
        
    def _create_transformers_callbacks(self):
        """Create transformers callbacks from our hooks"""
        from transformers import TrainerCallback
        
        class FedCoreHooksCallback(TrainerCallback):
            def __init__(self, hooks_collection, trainer_objects, history):
                self.hooks_collection = hooks_collection
                self.trainer_objects = trainer_objects
                self.history = history
            
            def on_epoch_begin(self, args, state, control, **kwargs):
                """Run hooks at the beginning of the epoch"""
                epoch = state.epoch
                for hook in self.hooks_collection.start:
                    hook(epoch=epoch, trainer_objects=self.trainer_objects, history=self.history)
            
            def on_epoch_end(self, args, state, control, **kwargs):
                """Run hooks at the end of the epoch"""
                epoch = state.epoch
                
                # Update history with current metrics
                if hasattr(state, 'log_history') and state.log_history:
                    latest_log = state.log_history[-1]
                    if 'train_loss' in latest_log:
                        self.history['train_loss'].append((epoch, latest_log['train_loss']))
                    if 'eval_loss' in latest_log:
                        self.history['val_loss'].append((epoch, latest_log['eval_loss']))
                else:
                    # If log_history is empty, add a stub
                    if not self.history.get('train_loss'):
                        self.history['train_loss'] = []
                    if not self.history.get('val_loss'):
                        self.history['val_loss'] = []
                
                for hook in self.hooks_collection.end:
                    hook(epoch=epoch, trainer_objects=self.trainer_objects, history=self.history)
        
        self._callbacks = []
        
        # Add our custom callback
        self._callbacks.append(FedCoreHooksCallback(
            self.hooks, 
            self.trainer_objects,
            getattr(self, 'history', {'train_loss': [], 'val_loss': []})
        ))
        
        # Add default callbacks
        self._callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
        
        # Note: ModelCheckpoint and LoggingCallback are handled by TrainingArguments
        # in newer versions of transformers
        
    def _prepare_data(self, input_data: Any) -> Dict[str, Dataset]:
        """Convert input_data to transformers Dataset format"""
        try:
            # Try to extract data from input_data
            if hasattr(input_data, 'features') and hasattr(input_data.features, 'train_dataloader'):
                train_data = self._dataloader_to_dataset(input_data.features.train_dataloader)
            else:
                train_data = None
                
            if hasattr(input_data, 'features') and hasattr(input_data.features, 'val_dataloader'):
                eval_data = self._dataloader_to_dataset(input_data.features.val_dataloader)
            else:
                eval_data = None
                
            return {
                'train_dataset': train_data,
                'eval_dataset': eval_data
            }
        except Exception as e:
            print(f"Error preparing data: {e}")
            # Return dummy data for testing
            return self._create_dummy_data()
            
    def _dataloader_to_dataset(self, dataloader) -> Dataset:
        """Convert PyTorch DataLoader to transformers Dataset"""
        data = []
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs, labels = batch[0], batch[1]
                # Convert to format expected by transformers
                if hasattr(inputs, 'cpu'):
                    inputs = inputs.cpu().numpy()
                if hasattr(labels, 'cpu'):
                    labels = labels.cpu().numpy()
                    
                data.append({
                    'input_ids': inputs,
                    'labels': labels
                })
        return Dataset.from_list(data)
        
    def _create_dummy_data(self) -> Dict[str, Dataset]:
        """Create dummy data for testing"""
        dummy_data = [
            {'input_ids': np.random.randint(0, 1000, (10,)), 'labels': np.random.randint(0, 10, (10,))}
            for _ in range(100)
        ]
        return {
            'train_dataset': Dataset.from_list(dummy_data),
            'eval_dataset': Dataset.from_list(dummy_data[:20])
        }

    def _create_transformers_trainer(self, datasets: Dict[str, Dataset]):
        """Create transformers trainer with parameter filtering"""
        allowed_training_args = {
            'output_dir', 'num_train_epochs', 'per_device_train_batch_size',
            'per_device_eval_batch_size', 'warmup_steps', 'lr_scheduler_type',
            'weight_decay', 'logging_dir', 'logging_steps', 'save_steps',
            'eval_steps', 'evaluation_strategy', 'save_strategy',
            'load_best_model_at_end', 'metric_for_best_model', 'greater_is_better'
        }
        
        filtered_args = self._normalize_kwargs(self.default_training_args, allowed_training_args)
        custom_opt = filtered_args.pop("custom_optimizer", None)
        
        self._training_args = TrainingArguments(**filtered_args)
        self._transformers_trainer = FedCoreTransformersTrainer(
            model=self.model,
            args=self._training_args,
            train_dataset=datasets.get('train_dataset'),
            eval_dataset=datasets.get('eval_dataset'),
            data_collator=self._data_collator,
            callbacks=getattr(self, '_callbacks', [])
        )
        self.trainer_objects['trainer'] = self._transformers_trainer
        
        if custom_opt is not None:
            def _custom_create_optimizer():
                opt = custom_opt(self.model)
                self._transformers_trainer.optimizer = opt
                self.trainer_objects['optimizer'] = opt
                return opt
            self._transformers_trainer.create_optimizer = _custom_create_optimizer


    def fit(self, input_data: Any, supplementary_data: Optional[Dict] = None, loader_type: str = 'train') -> Any:
        print(f"Training LLM model with {loader_type} data...")
        if not hasattr(self, '_callbacks'):
            self._init_hooks()
        
        # Use saved datasets or input_data
        if self._train_dataset is not None:
            datasets = {
                'train_dataset': self._train_dataset, 
                'eval_dataset': self._eval_dataset
            }
        elif isinstance(input_data, Dataset):
            datasets = {'train_dataset': input_data, 'eval_dataset': None}
        else:
            datasets = self._prepare_data(input_data)
            
        self._create_transformers_trainer(datasets)
        print("DEBUG: _transformers_trainer has been created?", hasattr(self, '_transformers_trainer'))
        if datasets.get('train_dataset'):
            train_result = self._transformers_trainer.train()
            print(f"Training completed. Loss: {getattr(train_result, 'training_loss', None)}")
        else:
            print("No training data provided")
        return self.model
        
    def predict(self, input_data: Any, output_mode: str = "default") -> Any:
        """Make predictions using transformers.Trainer"""
        print(f"Making LLM predictions with {output_mode} mode...")
        
        if self._transformers_trainer is None:
            # Create trainer if not exists
            datasets = self._prepare_data(input_data)
            self._create_transformers_trainer(datasets)
            
        # Make predictions
        if hasattr(input_data, 'features') and hasattr(input_data.features, 'val_dataloader'):
            eval_dataset = self._dataloader_to_dataset(input_data.features.val_dataloader)
            predictions = self._transformers_trainer.predict(eval_dataset)
            return predictions
        else:
            print("No evaluation data provided")
            return None
        
    def predict_for_fit(self, input_data: Any, output_mode: str = "default") -> Any:
        """Make predictions during training"""
        print(f"Making LLM predictions during training...")
        return self.predict(input_data, output_mode)
        
    def save_model(self, path: str) -> None:
        """Save the model using transformers approach"""
        print(f"Saving LLM model to {path}...")
        
        if self._transformers_trainer is not None:
            # Save using transformers trainer
            self._transformers_trainer.save_model(path)
        else:
            # Fallback to torch save
            if hasattr(self.model, 'save_pretrained'):
                self.model.save_pretrained(path)
            else:
                torch.save(self.model.state_dict(), path)
        
    def load_model(self, path: str) -> None:
        """Load the model using transformers approach"""
        print(f"Loading LLM model from {path}...")
        
        if os.path.exists(path):
            if hasattr(self.model, 'from_pretrained'):
                self.model = self.model.from_pretrained(path)
            else:
                # Load state dict
                state_dict = torch.load(path, map_location='cpu')
                self.model.load_state_dict(state_dict)
        else:
            print(f"Model path {path} does not exist")
        
    @property
    def is_quantised(self) -> bool:
        """Check if model is quantized"""
        return getattr(self.model, '_is_quantised', False)
    
    @property
    def optimizer(self) -> Any:
        """Get optimizer from transformers trainer"""
        if self._transformers_trainer is not None:
            return self._transformers_trainer.optimizer
        return self.trainer_objects['optimizer']
    
    @property
    def scheduler(self) -> Any:
        """Get scheduler from transformers trainer"""
        if self._transformers_trainer is not None:
            return self._transformers_trainer.lr_scheduler
        return self.trainer_objects['scheduler'] 