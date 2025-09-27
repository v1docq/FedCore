"""
LLM Trainer implementation using transformers library
Real integration with transformers.Trainer
"""
import torch
from typing import Any, Dict, Optional, Iterable, Union
from enum import Enum
from tqdm import tqdm


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
from fedot.core.data.data import OutputData
from fedot.core.repository.dataset_types import DataTypesEnum

from fedcore.api.utils.data import DataLoaderHandler
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
    
    def _convert_predict(self, predictions: torch.Tensor, output_mode: str = "default") -> torch.Tensor:
        """Convert predictions to appropriate format"""
        if isinstance(predictions, torch.Tensor):
            return predictions
        
        return torch.tensor(predictions)

    def _create_trainer(self, datasets: Dict[str, Dataset]):
        """Create transformers trainer with FedCore integration"""
        allowed_training_args = {
            'output_dir', 'num_train_epochs', 'per_device_train_batch_size',
            'per_device_eval_batch_size', 'warmup_steps', 'lr_scheduler_type',
            'weight_decay', 'logging_dir', 'logging_steps', 'save_steps',
            'eval_steps', 'evaluation_strategy', 'save_strategy',
            'load_best_model_at_end', 'metric_for_best_model', 'greater_is_better'
        }
        
        filtered_args = self._normalize_kwargs(self.default_training_args, allowed_training_args)
        
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
        
        
        has_val_loader = hasattr(input_data, 'features') and hasattr(input_data.features, 'val_dataloader')
        if self._trainer is None and has_val_loader:
            datasets = self._prepare_data(input_data)
            self._create_trainer(datasets)

        if self._trainer is not None and has_val_loader:
            eval_dataset = self._dataloader_to_dataset(input_data.features.val_dataloader)
            return self._trainer.predict(eval_dataset)

        predictions_output = self._predict_model(input_data, output_mode)
        pred_values = torch.tensor(predictions_output.predictions)
        
        output_data = OutputData(
            idx=torch.arange(len(pred_values)),
            task=input_data.task,
            predict=pred_values,
            target=None,
            data_type=DataTypesEnum.table,
        )
        
        return output_data
        
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
    
    @torch.no_grad()
    def _predict_model(
            self, x_test: Union[CompressionInputData, InputData], output_mode: str = "default"
    ):
        model: torch.nn.Module = self.model or x_test.target
        model.eval()
        prediction = []
        
        dataloader = DataLoaderHandler.check_convert(
            getattr(x_test, 'test_dataloader', None) or getattr(x_test, 'val_dataloader', None),
            mode=self.batch_type,
            max_batches=self.calib_batch_limit
        )
        
        if self.task_type is None:
            self.task_type = x_test.task.task_type
            
        for i, batch in tqdm(enumerate(dataloader, 1), total=len(dataloader)):
            if isinstance(batch, dict):
                inputs_dict = {}
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
                inputs_embeds = batch.get('inputs_embeds')

                if input_ids is not None:
                    inputs_dict['input_ids'] = input_ids.to(self.device)
                    if attention_mask is not None:
                        inputs_dict['attention_mask'] = attention_mask.to(self.device)
                elif inputs_embeds is not None:
                    inputs_dict['inputs_embeds'] = inputs_embeds.to(self.device)
                    if attention_mask is not None:
                        inputs_dict['attention_mask'] = attention_mask.to(self.device)

                pred = model(**inputs_dict) if len(inputs_dict) > 0 else model()
            else:
                seq = list(batch)
                if len(seq) >= 2 and hasattr(seq[-1], 'dtype'):
                    seq_inputs = seq[:-1]
                else:
                    seq_inputs = seq

                inputs_dict = {}
                if len(seq_inputs) >= 1 and hasattr(seq_inputs[0], 'dtype'):
                    t0 = seq_inputs[0].to(self.device)
                    if t0.dtype in (torch.int32, torch.int64):
                        inputs_dict['input_ids'] = t0
                    else:
                        inputs_dict['inputs_embeds'] = t0
                if len(seq_inputs) >= 2 and hasattr(seq_inputs[1], 'dtype'):
                    t1 = seq_inputs[1].to(self.device)
                    if 'inputs_embeds' not in inputs_dict and t1.dtype in (torch.int32, torch.int64, torch.bool):
                        inputs_dict['attention_mask'] = t1
                
                pred = model(**inputs_dict)
            
            pred_tensor = getattr(pred, 'logits', pred)
            prediction.append(pred_tensor)
            
            if i % getattr(self, '_clear_each', 10) == 0:
                self._clear_cache()
                
        return self._convert_predict(torch.cat(prediction), output_mode)
    
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