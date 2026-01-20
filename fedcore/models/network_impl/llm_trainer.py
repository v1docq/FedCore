"""
LLM Trainer implementation using transformers library
Real integration with transformers.Trainer
"""
import torch
import logging
from typing import Any, Dict, Optional, Iterable, Union, List
from enum import Enum
from tqdm import tqdm
import logging 


# Transformers imports
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback
from datasets import Dataset
import numpy as np
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum

from fedcore.api.utils.data import DataLoaderHandler
from fedcore.data.data import CompressionInputData, CompressionOutputData
from fedcore.models.network_impl.utils._base import BaseTrainer
from fedcore.models.network_impl.utils.hooks_collection import HooksCollection
from fedcore.models.network_impl.utils.hooks import LoggingHooks, ModelLearningHooks, OptimizerGen, SchedulerRenewal


class FedCoreTransformersTrainer(TrainerCallback):
    """
    Transformers Callback with FedCore hooks integration.
    
    Combines HuggingFace Transformers callbacks with FedCore's hook system for
    OptimizerGen, SchedulerRenewal, and other training hooks.
    """
    
    def __init__(
        self,
        model=None,
        hooks_collection: Optional[HooksCollection] = None,
        hooks_params: Optional[Dict] = None,
        additional_hooks: Optional[Iterable[Enum]] = None,
    ):
        super().__init__()
        self.model = model
        self.hooks_collection = hooks_collection or HooksCollection()
        self.hooks_params = hooks_params or {}
        self._hooks = [LoggingHooks, ModelLearningHooks]
        if additional_hooks:
            self._hooks.extend(additional_hooks)
        
        self.trainer_objects = {
            'model': model,
            'optimizer': None,
            'scheduler': None,
            'stop': False
        }
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        if self.hooks_params:
            self._init_hooks()
    
    def _init_hooks(self):
        """Initialize FedCore hooks based on parameters"""
        for hook_enum in self._hooks:
            for hook_elem in hook_enum:
                hook_class = hook_elem.value
                if hook_class.check_init(self.hooks_params):
                    hook_instance = hook_class(self.hooks_params, self.model)
                    self.hooks_collection.append(hook_instance)

    def on_epoch_begin(self, args: TrainingArguments, state, control, **kwargs):
        """Execute FedCore hooks at the beginning of each epoch"""
        epoch = state.epoch if hasattr(state, 'epoch') else 0
        trainer = kwargs.get('trainer')
        
        if trainer:
            if hasattr(trainer, 'optimizer') and trainer.optimizer:
                self.trainer_objects['optimizer'] = trainer.optimizer
            if hasattr(trainer, 'lr_scheduler') and trainer.lr_scheduler:
                self.trainer_objects['scheduler'] = trainer.lr_scheduler
        
        for hook in self.hooks_collection.start:
            hook(epoch, 
                 trainer_objects=self.trainer_objects,
                 history=self.history)
        
        if trainer:
            if self.trainer_objects.get('optimizer'):
                trainer.optimizer = self.trainer_objects['optimizer']
            if self.trainer_objects.get('scheduler'):
                trainer.lr_scheduler = self.trainer_objects['scheduler']
        
        if self.trainer_objects.get('stop', False): 
            control.should_training_stop = True
        
        return control
    
    def on_epoch_end(self, args: TrainingArguments, state, control, **kwargs):
        """Execute FedCore hooks at the end of each epoch"""
        epoch = state.epoch if hasattr(state, 'epoch') else 0
        trainer = kwargs.get('trainer')
        
        if hasattr(state, 'log_history') and state.log_history:
            latest_log = state.log_history[-1]
            if 'loss' in latest_log:
                self.history['train_loss'].append((epoch, latest_log['loss']))
            if 'eval_loss' in latest_log:
                self.history['val_loss'].append((epoch, latest_log['eval_loss']))
        
        if trainer:
            if hasattr(trainer, 'optimizer') and trainer.optimizer:
                self.trainer_objects['optimizer'] = trainer.optimizer
            if hasattr(trainer, 'lr_scheduler') and trainer.lr_scheduler:
                self.trainer_objects['scheduler'] = trainer.lr_scheduler
        
        val_loader = None
        criterion = None
        if trainer:
            if hasattr(trainer, 'eval_dataset') and trainer.eval_dataset:
                val_loader = trainer.get_eval_dataloader(trainer.eval_dataset)
            if hasattr(trainer, 'compute_loss'):
                criterion = trainer.compute_loss
        
        for hook in self.hooks_collection.end:
            hook(epoch,
                 trainer_objects=self.trainer_objects,
                 history=self.history,
                 val_loader=val_loader,
                 criterion=criterion)
        
        if trainer:
            if self.trainer_objects.get('optimizer'):
                trainer.optimizer = self.trainer_objects['optimizer']
            if self.trainer_objects.get('scheduler'):
                trainer.lr_scheduler = self.trainer_objects['scheduler']
        
        if self.trainer_objects.get('stop', False):
            control.should_training_stop = True
        
        return control
    
    def on_step_end(self, args, state, control, **kwargs):
        """Track learning rate and sync trainer objects on each step"""
        trainer = kwargs.get('trainer')
        if trainer:
            if hasattr(trainer, 'optimizer') and trainer.optimizer:
                self.trainer_objects['optimizer'] = trainer.optimizer
                if hasattr(trainer.optimizer, 'param_groups'):
                    current_lr = trainer.optimizer.param_groups[0]['lr']
                    current_step = state.global_step if hasattr(state, 'global_step') else 0
                    self.history['learning_rates'].append((current_step, current_lr))
            
            if hasattr(trainer, 'lr_scheduler') and trainer.lr_scheduler:
                self.trainer_objects['scheduler'] = trainer.lr_scheduler
                
        return control
    
    def on_init_end(self, args, state, control, **kwargs):
        """Called after Trainer initialization - setup optimizer and scheduler via hooks"""
        trainer = kwargs.get('trainer')
        if trainer:
            self.trainer_objects['trainer'] = trainer
            self.trainer_objects['model'] = trainer.model
            
        return control


class LLMTrainer(BaseTrainer):
    """
    LLM Trainer that implements our interfaces with real transformers.Trainer integration
    """
    
    DEFAULT_TRAINING_ARGS = {
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
            'no_cuda': False,
        }
    
    DEFAULT_GENERATION_PARAMS = {
        'max_length': 100,
        'max_new_tokens': None,
        'temperature': 1.0,
        'top_k': 50,
        'top_p': 1.0,
        'do_sample': True,
        'num_beams': 1,
        'early_stopping': False,
        'no_repeat_ngram_size': 0,
        'repetition_penalty': 1.0,
        'length_penalty': 1.0,
    }
        
    ALLOWED_TRAINING_ARGS = {
        'output_dir', 'num_train_epochs', 'per_device_train_batch_size',
        'per_device_eval_batch_size', 'warmup_steps', 'lr_scheduler_type',
        'weight_decay', 'logging_dir', 'logging_steps', 'save_steps',
        'eval_steps', 'evaluation_strategy', 'save_strategy',
        'load_best_model_at_end', 'metric_for_best_model', 'greater_is_better',
        'no_cuda'
    }
    
    def __init__(self, params: Optional[Dict] = None, **kwargs):
        # if model is None and params and isinstance(params.get('custom_learning_params'), dict):
        #     nested = params.get('custom_learning_params')
        #     model = nested.get('model', model)
        
        super().__init__(params=params)
        self.model = self.params.get("model", None)
        self.tokenizer = self.params.get("tokenizer", None)
        
        self.default_training_args = self.DEFAULT_TRAINING_ARGS.copy()
        if params:
            training_params = {k: v for k, v in params.items() 
                             if k not in ['model', 'tokenizer', 'custom_learning_params']}
            self.default_training_args.update(training_params)
        
        # Initialize default generation parameters
        self.default_generation_params = self.DEFAULT_GENERATION_PARAMS.copy()
        if params:
            generation_params = {
                k: v for k, v in params.items() 
                if k in self.DEFAULT_GENERATION_PARAMS
            }
            self.default_generation_params.update(generation_params)
        
        # Initialize calibration batch limit (used in prediction)
        self.calib_batch_limit = self.learning_params.get('calib_batch_limit', None)
        self.batch_type = self.learning_params.get('batch_type', None)  # None means use default from DataLoaderHandler
        
        self._trainer = None
        self._training_args = None
        self._data_collator = None
        self._fedcore_callback = None
        self._hooks_initialized = False
        self.task_type = None
    
    @property
    def epochs(self) -> int:
        """Get the number of training epochs."""
        if self._training_args is not None:
            return int(self._training_args.num_train_epochs)
        return int(self.default_training_args.get('num_train_epochs', 3))
        
    def _init_hooks(self) -> None:
        """
        Initialize hooks for the model.
        
        Marks hooks as ready to be initialized in FedCoreTransformersTrainer.
        Actual hook creation happens when FedCoreTransformersTrainer is instantiated.
        """
        if self._hooks_initialized:
            return
        
        self._hooks_initialized = True
        
    def _prepare_data(self, input_data: Union[InputData, CompressionInputData]) -> Dict[str, Dataset]:
        """Convert InputData/CompressionInputData to transformers Dataset format"""
        train_data = self._dataloader_to_dataset(input_data.train_dataloader)
        eval_data = self._dataloader_to_dataset(input_data.val_dataloader)
            
        return {
            'train_dataset': train_data,
            'eval_dataset': eval_data
        }
            
    def _dataloader_to_dataset(self, dataloader) -> Dataset:
        """
        Convert DataLoader to HuggingFace Dataset.
        
        Dataset reinstantiation is needed because:
        - Transformers Trainer requires HuggingFace Dataset, not PyTorch DataLoader
        - DataLoader is stateful (iterator) and incompatible with Transformers' internal data handling
        - Dataset format allows Transformers to manage batching, shuffling, and distributed training
        - Converts from our batch formats (tuples/dicts) to standard Dataset.__getitem__ format
        """
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
        if self.model is None and self._trainer is not None:
            self.model = self._trainer.model
        
        if self.model is None:
            raise ValueError("LLMTrainer initialization failed: model is None after parameter resolution. Ensure 'model' or 'initial_assumption' is provided in config (including inside 'custom_learning_params').")

        if torch.cuda.is_available():
            device = torch.device('cuda')
            self.model = self.model.to(device)

        if not self._hooks_initialized:
            self._init_hooks()
        
        filtered_args = self._normalize_kwargs(self.default_training_args, self.ALLOWED_TRAINING_ARGS)
        self._training_args = TrainingArguments(**filtered_args)
        
        hooks_params = self.default_training_args.copy()
        if self.params:
            for key in ['optimizer', 'scheduler', 'criterion', 'learning_rate']:
                if key in self.params:
                    hooks_params[key] = self.params[key]
        
        self._fedcore_callback = FedCoreTransformersTrainer(
            model=self.model,
            hooks_params=hooks_params,
            additional_hooks=self._additional_hooks
        )
        
        callbacks = [self._fedcore_callback, EarlyStoppingCallback(early_stopping_patience=3)]
        

        self._trainer = Trainer(
            model=self.model,
            args=self._training_args,
            train_dataset=datasets.get('train_dataset'),
            eval_dataset=datasets.get('eval_dataset'),
            data_collator=self._data_collator,
            callbacks=callbacks
        )
        
        self.trainer_objects['trainer'] = self._trainer

    def fit(self, input_data: CompressionInputData, 
            supplementary_data: Optional[Dict] = None, loader_type: str = 'train') -> Any:
        """
        Train the model using InputData/CompressionInputData.
        
        Input can be either InputData (FEDOT) or CompressionInputData (FedCore).
        """
        logging.info(f"Training LLM model with {loader_type} data...")
        
        # Final fallback: pull model from input_data if still missing
        if self.model is None:
            candidate_model = getattr(input_data, 'target', None)
            if candidate_model is None and hasattr(input_data, 'target'):
                candidate_model = input_data.target
            if candidate_model is not None:
                self.model = candidate_model

        datasets = self._prepare_data(input_data)
        self._create_trainer(datasets)
        # self.execute_hooks('start', epoch=0)
        
        train_result = self._trainer.train()
        logging.info(f"Training completed. Loss: {getattr(train_result, 'training_loss', None)}")
        if self._fedcore_callback:
            self.history.update(self._fedcore_callback.history)
        
        self.model = self._trainer.model
        
        return self.model
        
        
    def generate_long_text(
        self,
        input_data: Union[CompressionInputData, InputData],
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        num_beams: Optional[int] = None,
        early_stopping: Optional[bool] = None,
        no_repeat_ngram_size: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        length_penalty: Optional[float] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **generation_kwargs
    ) -> List[str]:
        """
        Generate long text sequences for input prompts.
        
        Args:
            input_data: Input data with prompts (via dataloader)
            max_length: Maximum length of generated sequence (including prompt)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random output)
            top_k: Top-k sampling - limit selection to top-k tokens
            top_p: Nucleus sampling - limit by cumulative probability
            do_sample: Whether to use sampling (True) or greedy decoding (False)
            num_beams: Number of beams for beam search
            early_stopping: Whether to stop beam search when EOS is reached
            no_repeat_ngram_size: Size of n-gram that cannot be repeated
            repetition_penalty: Penalty for repetitions (1.0 = no penalty)
            length_penalty: Length penalty for beam search
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            **generation_kwargs: Additional parameters for model.generate()
            
        Returns:
            List[str]: List of generated texts
        """
        if self.model is None:
            raise ValueError("Model is not set. Cannot generate text.")
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not set. Cannot generate text.")
        
        model = self.model
        tokenizer = self.tokenizer
        model.eval()
        
        # Use default parameters if not explicitly provided
        gen_params = self.default_generation_params.copy()
        gen_params.update({k: v for k, v in {
            'max_length': max_length,
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'do_sample': do_sample,
            'num_beams': num_beams,
            'early_stopping': early_stopping,
            'no_repeat_ngram_size': no_repeat_ngram_size,
            'repetition_penalty': repetition_penalty,
            'length_penalty': length_penalty,
        }.items() if v is not None})
        
        gen_params.update(generation_kwargs)

        # Remove temperature when do_sample=False (greedy decoding)
        if gen_params.get('do_sample') is False:
            gen_params.pop('temperature', None)
        
        # Remove max_length if max_new_tokens is set (to avoid warning)
        if gen_params.get('max_new_tokens') is not None:
            gen_params.pop('max_length', None)
        
        # Get pad_token_id and eos_token_id from tokenizer if not provided
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, 'pad_token_id', None)
            if pad_token_id is None:
                pad_token_id = getattr(tokenizer, 'eos_token_id', None)
        
        if eos_token_id is None:
            eos_token_id = getattr(tokenizer, 'eos_token_id', None)
        
        if pad_token_id is not None:
            gen_params['pad_token_id'] = pad_token_id
        if eos_token_id is not None:
            gen_params['eos_token_id'] = eos_token_id
        
        generated_texts = []
        
        # Get dataloader
        dataloader = DataLoaderHandler.check_convert(
            getattr(input_data, 'test_dataloader', None) or 
            getattr(input_data, 'val_dataloader', None) or
            getattr(input_data, 'train_dataloader', None),
            mode=self.batch_type,
            max_batches=self.calib_batch_limit
        )
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating text"):
                # Prepare input data - batch should already be a dict from DataLoader
                if isinstance(batch, dict):
                    input_ids = batch.get('input_ids')
                    attention_mask = batch.get('attention_mask')
                elif isinstance(batch, (list, tuple)):
                    # Handle tuple/list format: (input_ids, attention_mask) or (dict, ...)
                    if len(batch) > 0:
                        first_item = batch[0]
                        if isinstance(first_item, dict):
                            # Nested dict in tuple/list
                            input_ids = first_item.get('input_ids')
                            attention_mask = first_item.get('attention_mask')
                        elif isinstance(first_item, torch.Tensor):
                            # Direct tensor
                            input_ids = first_item
                            attention_mask = batch[1] if len(batch) > 1 and isinstance(batch[1], torch.Tensor) else None
                        else:
                            input_ids = first_item
                            attention_mask = batch[1] if len(batch) > 1 else None
                    else:
                        input_ids = None
                        attention_mask = None
                else:
                    # Single tensor
                    input_ids = batch if isinstance(batch, torch.Tensor) else None
                    attention_mask = None
                
                if input_ids is None:
                    continue
                
                # Ensure input_ids is a tensor
                if not isinstance(input_ids, torch.Tensor):
                    # Try to convert to tensor
                    if isinstance(input_ids, (list, tuple)):
                        input_ids = torch.tensor(input_ids, device=self.device)
                    elif isinstance(input_ids, dict):
                        # If it's still a dict, something is wrong, skip this batch
                        logging.warning(f"Unexpected dict structure in input_ids: {type(input_ids)}")
                        continue
                    else:
                        try:
                            input_ids = torch.tensor(input_ids, device=self.device)
                        except (TypeError, ValueError) as e:
                            logging.warning(f"Cannot convert input_ids to tensor: {e}")
                            continue
                
                # Move to device if not already there
                if input_ids.device != self.device:
                    input_ids = input_ids.to(self.device)
                    
                if attention_mask is not None:
                    if not isinstance(attention_mask, torch.Tensor):
                        if isinstance(attention_mask, (list, tuple)):
                            attention_mask = torch.tensor(attention_mask, device=self.device)
                        else:
                            try:
                                attention_mask = torch.tensor(attention_mask, device=self.device)
                            except (TypeError, ValueError):
                                attention_mask = None
                    elif attention_mask.device != self.device:
                        attention_mask = attention_mask.to(self.device)
                
                # Prepare arguments for generate
                generate_kwargs = {'input_ids': input_ids}
                if attention_mask is not None:
                    generate_kwargs['attention_mask'] = attention_mask
                generate_kwargs.update(gen_params)
                
                # Generation
                try:
                    generated_ids = model.generate(**generate_kwargs)
                except Exception as e:
                    logging.warning(f"Generation failed for batch: {e}")
                    # Fallback: try without attention_mask
                    if 'attention_mask' in generate_kwargs:
                        del generate_kwargs['attention_mask']
                        generated_ids = model.generate(**generate_kwargs)
                    else:
                        raise
                
                # Decoding
                # If input_ids were in batch, extract only new tokens
                if gen_params.get('max_new_tokens') is not None or gen_params.get('max_length') is not None:
                    # Extract only generated part
                    generated_only = generated_ids[:, input_ids.shape[1]:]
                else:
                    generated_only = generated_ids
                
                # Decode each element in batch
                batch_texts = tokenizer.batch_decode(
                    generated_only, 
                    skip_special_tokens=True
                )
                generated_texts.extend(batch_texts)
        
        return generated_texts

    def predict_for_fit(self, input_data: CompressionInputData,  
                       output_mode: str = "default",
                       max_length: Optional[int] = None,
                       max_new_tokens: Optional[int] = None,
                       temperature: Optional[float] = None,
                       top_k: Optional[int] = None,
                       top_p: Optional[float] = None,
                       do_sample: Optional[bool] = None,
                       num_beams: Optional[int] = None,
                       **generation_kwargs) -> Any:
        """Make predictions during training"""
        return self.predict(input_data, output_mode, max_length, max_new_tokens,
                          temperature, top_k, top_p, do_sample, num_beams, **generation_kwargs)
    
    def predict(self, input_data: CompressionInputData,  
                output_mode: str = "default",
                max_length: Optional[int] = None,
                max_new_tokens: Optional[int] = None,
                temperature: Optional[float] = None,
                top_k: Optional[int] = None,
                top_p: Optional[float] = None,
                do_sample: Optional[bool] = None,
                num_beams: Optional[int] = None,
                **generation_kwargs) -> Any:
        """
        Make predictions using InputData/CompressionInputData.
        
        Args:
            input_data: Input data
            output_mode: Output mode - "default" (logits), "texts" (text generation),
                        "labels", "probs", "raw", "compress"
            max_length: Maximum length for generation (only for output_mode="texts")
            max_new_tokens: Maximum number of new tokens (only for output_mode="texts")
            temperature: Generation temperature (only for output_mode="texts")
            top_k: Top-k sampling (only for output_mode="texts")
            top_p: Nucleus sampling (only for output_mode="texts")
            do_sample: Use sampling (only for output_mode="texts")
            num_beams: Number of beams for beam search (only for output_mode="texts")
            **generation_kwargs: Additional generation parameters
            
        Returns:
            CompressionOutputData with predictions
        """
        
        # If text generation mode, use generate_long_text
        if output_mode == "texts":
            generated_texts = self.generate_long_text(
                input_data=input_data,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                num_beams=num_beams,
                **generation_kwargs
            )
            
            # Create CompressionOutputData with texts
            extracted_fields = self._extract_output_fields(input_data)
            
            if self.task_type is None and input_data is not None:
                if hasattr(input_data, 'task'):
                    self.task_type = input_data.task.task_type if hasattr(input_data.task, 'task_type') else input_data.task
                elif hasattr(input_data, 'features') and hasattr(input_data.features, 'task'):
                    task_obj = input_data.features.task
                    self.task_type = task_obj.task_type if hasattr(task_obj, 'task_type') else task_obj
            
            checkpoint_info = self._register_model_checkpoint(
                model=self.model,
                stage='after'
            )
            
            output_data = CompressionOutputData(
                features=extracted_fields['features'],
                task=self.task_type,
                predict=generated_texts,  # List of strings
                num_classes=extracted_fields['num_classes'],
                train_dataloader=extracted_fields['train_dataloader'],
                val_dataloader=extracted_fields['val_dataloader'],
                data_type=DataTypesEnum.table,
                model=self.model,
                checkpoint_path=checkpoint_info['checkpoint_path'],
                model_id=checkpoint_info['model_id'],
                fedcore_id=checkpoint_info['fedcore_id'],
            )
            
            return output_data
        
        # Otherwise use standard prediction logic
        has_val_loader = hasattr(input_data, 'val_dataloader')

        if self._trainer is not None and has_val_loader:
            if torch.cuda.is_available():
                device = torch.device('cuda')
                self._trainer.model = self._trainer.model.to(device)
                for buffer in self._trainer.model.buffers():
                    buffer.data = buffer.data.to(device)
            self._trainer.model.eval()
            
            eval_dataset = self._dataloader_to_dataset(input_data.val_dataloader)
            return self._trainer.predict(eval_dataset)
        elif self._trainer is None and has_val_loader:
            if self.model is None:
                raise ValueError("Cannot create trainer for prediction: model is None. Call fit() first or provide model in initialization.")
            datasets = self._prepare_data(input_data)
            self._create_trainer(datasets)
            eval_dataset = self._dataloader_to_dataset(input_data.val_dataloader)
            return self._trainer.predict(eval_dataset)

        predictions_output = self._predict_model(input_data, output_mode)
        pred_values = torch.tensor(predictions_output.predictions)
        
        output_data = CompressionOutputData(
            # idx=torch.arange(len(pred_values)),
            task=input_data.task,
            predict=pred_values,
            target=None,
            data_type=DataTypesEnum.table,
        )
        
        return output_data
        
    def save_model(self, path: str) -> None:
        """Save the model using transformers approach"""
        logging.info(f"Saving LLM model to {path}...")
        
        if self._trainer is not None:
            self._trainer.save_model(path)
        else:
            super().save_model(path)
        
    def load_model(self, path: str) -> None:
        """Load the model using transformers approach"""
        logging.info(f"Loading LLM model from {path}...")

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
            if hasattr(x_test, 'task'):
                self.task_type = x_test.task.task_type if hasattr(x_test.task, 'task_type') else x_test.task
            elif hasattr(x_test, 'features') and hasattr(x_test.features, 'task'):
                task_obj = x_test.features.task
                self.task_type = task_obj.task_type if hasattr(task_obj, 'task_type') else task_obj
            
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
                
        return self._convert_predict(torch.cat(prediction), output_mode, x_test)
    
    def _convert_predict(self, pred: Union[torch.Tensor, np.ndarray], output_mode: str = "default", 
                         input_data: Union[CompressionInputData, InputData] = None) -> CompressionOutputData:
        """Convert predictions to CompressionOutputData format"""
        if isinstance(pred, torch.Tensor):
            pred_values = pred.cpu().detach()
        elif isinstance(pred, np.ndarray):
            pred_values = torch.from_numpy(pred)
        elif isinstance(pred, list):
            pred_values = torch.tensor(pred)
        else:
            try:
                pred_values = torch.tensor(pred)
            except (TypeError, ValueError) as e:
                raise TypeError(f"Prediction conversion failed: cannot convert {type(pred).__name__} to Tensor. Error: {e}")
        
        extracted_fields = self._extract_output_fields(input_data)
        
        if self.task_type is None and input_data is not None:
            if hasattr(input_data, 'task'):
                self.task_type = input_data.task.task_type if hasattr(input_data.task, 'task_type') else input_data.task
            elif hasattr(input_data, 'features') and hasattr(input_data.features, 'task'):
                task_obj = input_data.features.task
                self.task_type = task_obj.task_type if hasattr(task_obj, 'task_type') else task_obj
        
        checkpoint_info = self._register_model_checkpoint(
            model=self.model,
            stage='after'
        )
        
        predict = CompressionOutputData(
            features=extracted_fields['features'],
            task=self.task_type,
            predict=pred_values,
            num_classes=extracted_fields['num_classes'],
            train_dataloader=extracted_fields['train_dataloader'],
            val_dataloader=extracted_fields['val_dataloader'],
            data_type=DataTypesEnum.table,
            model=self.model,
            checkpoint_path=checkpoint_info['checkpoint_path'],
            model_id=checkpoint_info['model_id'],
            fedcore_id=checkpoint_info['fedcore_id'],
        )
        
        return predict

    
    @property
    def scheduler(self) -> Any:
        """Get scheduler from transformers trainer"""
        if self._fedcore_callback and 'scheduler' in self._fedcore_callback.trainer_objects:
            return self._fedcore_callback.trainer_objects['scheduler']
        return None