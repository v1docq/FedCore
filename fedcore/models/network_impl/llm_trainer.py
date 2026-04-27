"""
LLM Trainer implementation using transformers library
Real integration with transformers.Trainer
"""
import torch
import torch.nn.functional as F
import logging
from typing import Any, Dict, Optional, Iterable, Union, List
from enum import Enum
from tqdm import tqdm 
from pymonad.maybe import Maybe
import copy
from itertools import chain
from accelerate.state import AcceleratorState

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
from fedcore.models.network_impl.utils.hooks import LOGGING_HOOKS, MODEL_LEARNING_HOOKS, OptimizerGen, SchedulerRenewal


class FedCoreTransformersTrainer(TrainerCallback):
    """
    Transformers Callback with FedCore hooks integration.
    
    Thin adapter that connects HuggingFace Transformers callbacks with FedCore's hook system.
    Uses hooks, history, and trainer_objects from base_trainer (LLMTrainer) to avoid duplication.
    """
    
    def __init__(
        self,
        model=None,
        base_trainer=None,
    ):
        super().__init__()
        self.model = model
        self.base_trainer = base_trainer
        self.reference_batch = None
        self.current_step = 0
        self._reference_batch_saved = False
    
    @property
    def hooks_collection(self):
        """Access hooks from base_trainer (HooksCollection)"""
        return self.base_trainer.hooks
    
    @property
    def history(self):
        """Access history from base_trainer"""
        if 'ref_loss' not in self.base_trainer.history:
            self.base_trainer.history['ref_loss'] = []
        return self.base_trainer.history
    
    @property
    def trainer_objects(self):
        """Access trainer_objects from base_trainer"""
        return self.base_trainer.trainer_objects
    
    def _sync_trainer_objects(self, trainer):
        """Sync optimizer and scheduler from trainer to trainer_objects"""
        if not trainer:
            return
        
        if hasattr(trainer, 'optimizer') and trainer.optimizer:
            self.trainer_objects['optimizer'] = trainer.optimizer
        if hasattr(trainer, 'lr_scheduler') and trainer.lr_scheduler:
            self.trainer_objects['scheduler'] = trainer.lr_scheduler
    
    def _apply_trainer_objects(self, trainer):
        """Apply optimizer and scheduler from trainer_objects back to trainer"""
        if not trainer:
            return
        
        if self.trainer_objects.get('optimizer'):
            trainer.optimizer = self.trainer_objects['optimizer']
        if self.trainer_objects.get('scheduler'):
            trainer.lr_scheduler = self.trainer_objects['scheduler']
    
    def _save_reference_batch(self, inputs):
        """Save the first batch as reference batch for loss computation"""
        if self._reference_batch_saved or inputs is None:
            return
        
        self.reference_batch = copy.deepcopy(inputs)
        
        self._reference_batch_saved = True
    
    def _compute_reference_loss(self, trainer, model):
        """Compute loss on the reference batch"""
        # if self.reference_batch is None or model is None:
        #     return None
        
        ref_batch = self.reference_batch
        device = next(model.parameters()).device

        if isinstance(ref_batch, list) and len(ref_batch) > 0 and isinstance(ref_batch[0], dict):
            was_training = model.training
            model.eval()
            batch_losses = []
            with torch.no_grad():
                for batch in ref_batch:
                    filtered = self.base_trainer._normalize_kwargs(batch, {'input_ids', 'attention_mask', 'labels'})
                    for key, value in filtered.items():
                        if torch.is_tensor(value):
                            filtered[key] = value.to(device)
                    if 'input_ids' not in filtered:
                        continue
                    outputs = model(**filtered)
                    if hasattr(outputs, 'loss'):
                        batch_losses.append(outputs.loss.item())
                    elif isinstance(outputs, dict) and 'loss' in outputs:
                        batch_losses.append(outputs['loss'].item())
                    elif hasattr(trainer, 'compute_loss') and trainer.compute_loss:
                        l = trainer.compute_loss(model, filtered, return_outputs=False)
                        batch_losses.append(l.item() if isinstance(l, torch.Tensor) else l)
            if was_training:
                model.train()
            if not batch_losses:
                return None
            avg_loss = sum(batch_losses) / len(batch_losses)
            return [avg_loss]

        if isinstance(ref_batch, dict):
            allowed_keys = {'input_ids', 'attention_mask', 'labels'}
            filtered_batch = self.base_trainer._normalize_kwargs(ref_batch, allowed_keys)
            
            for key, value in filtered_batch.items():
                if torch.is_tensor(value):
                    filtered_batch[key] = value.to(device)
            
            if 'input_ids' not in filtered_batch:
                logging.warning("input_ids not found in reference batch after filtering")
                return None
            
            ref_batch = filtered_batch
            
        elif isinstance(ref_batch, (list, tuple)):
            ref_batch = tuple(v.to(device) if torch.is_tensor(v) else v 
                            for v in ref_batch)
        else:
            logging.warning(f"Unsupported reference batch format: {type(ref_batch)}")
            return None
        was_training = model.training
        model.eval()
        if isinstance(ref_batch, dict):
            batch_size = ref_batch['input_ids'].shape[0]
        elif isinstance(ref_batch, (list, tuple)):
            batch_size = ref_batch[0].shape[0]
        else:
            batch_size = 1
        
        losses_per_element = []
        
        with torch.no_grad():
            for i in range(batch_size):
                if isinstance(ref_batch, dict):
                    single_element_batch = {
                        key: value[i:i+1] if torch.is_tensor(value) else value
                        for key, value in ref_batch.items()
                    }
                elif isinstance(ref_batch, (list, tuple)):
                    single_element_batch = tuple(
                        v[i:i+1] if torch.is_tensor(v) else v 
                        for v in ref_batch
                    )
                else:
                    single_element_batch = ref_batch[i:i+1]
                
                if isinstance(single_element_batch, dict):
                    outputs = model(**single_element_batch)
                elif isinstance(single_element_batch, (list, tuple)):
                    outputs = model(*single_element_batch)
                else:
                    outputs = model(single_element_batch)
                
                if hasattr(outputs, 'loss'):
                    element_loss = outputs.loss.item()
                elif isinstance(outputs, dict) and 'loss' in outputs:
                    element_loss = outputs['loss'].item()
                elif hasattr(trainer, 'compute_loss') and trainer.compute_loss:
                    element_loss = trainer.compute_loss(model, single_element_batch, return_outputs=False)
                    element_loss = element_loss.item() if isinstance(element_loss, torch.Tensor) else element_loss
                else:
                    element_loss = None
                
                if element_loss is not None:
                    losses_per_element.append(element_loss)
            
            if was_training:
                model.train()
            
            return losses_per_element if losses_per_element else None

    # def on_step_begin(self, args, state, control, **kwargs):
    #     if not self._reference_batch_saved:
    #         inputs = kwargs.get('inputs')
    #         if inputs is not None:
    #             self._save_reference_batch(inputs)
    #     return control
    
    # def on_train_begin(self, args, state, control, **kwargs):
    #     """Called at the beginning of training"""
    #     if not self._reference_batch_saved:
    #         logging.warning("Reference batch was not saved in on_init_end")
        
    #     return control

    def on_epoch_begin(self, args: TrainingArguments, state, control, **kwargs):
        """Execute FedCore hooks at the beginning of each epoch"""
        step = state.global_step if hasattr(state, 'global_step') else self.current_step
        trainer = kwargs.get('trainer')
        
        if hasattr(state, 'epoch'):
            epoch = int(state.epoch)
        else:
            epoch = 0
        
        self._sync_trainer_objects(trainer)
        
        for hook in self.hooks_collection.start():
            hook(epoch, 
                 trainer_objects=self.trainer_objects,
                 history=self.history)
        
        self._apply_trainer_objects(trainer)
        
        if self.trainer_objects.get('stop', False): 
            control.should_training_stop = True
        
        return control
    
    def on_epoch_end(self, args: TrainingArguments, state, control, **kwargs):
        """Execute FedCore hooks at the end of each epoch"""
        step = state.global_step if hasattr(state, 'global_step') else self.current_step
        trainer = kwargs.get('trainer')
        
        if hasattr(state, 'epoch'):
            epoch = int(state.epoch)
        else:
            epoch = 0
            if trainer and hasattr(trainer, 'get_train_dataloader'):
                try:
                    train_dataloader = trainer.get_train_dataloader()
                    steps_per_epoch = len(train_dataloader) if train_dataloader else 1
                    epoch = step // steps_per_epoch if steps_per_epoch > 0 else 0
                except:
                    epoch = 0
        
        if hasattr(state, 'log_history') and state.log_history:
            latest_log = state.log_history[-1]
            if 'loss' in latest_log:
                self.history['train_loss'].append((epoch, latest_log['loss']))
            if 'eval_loss' in latest_log:
                self.history['val_loss'].append((epoch, latest_log['eval_loss']))
        
        self._sync_trainer_objects(trainer)
        
        val_loader = trainer.get_eval_dataloader(trainer.eval_dataset) if trainer and hasattr(trainer, 'eval_dataset') and trainer.eval_dataset else None
        criterion = trainer.compute_loss if trainer and hasattr(trainer, 'compute_loss') else None
        
        for hook in self.hooks_collection.end():
            hook(epoch, trainer_objects=self.trainer_objects, history=self.history, val_loader=val_loader, criterion=criterion)
        
        self._apply_trainer_objects(trainer)
        
        if self.trainer_objects.get('stop', False):
            control.should_training_stop = True
        
        return control
    
    def on_step_end(self, args, state, control, **kwargs):
        """Handle all training step logic: sync objects, track LR, compute reference loss"""
        trainer = kwargs.get('trainer')
        model = kwargs.get('model')  
        if not trainer:
            if self.base_trainer and hasattr(self.base_trainer, '_trainer'):
                trainer = self.base_trainer._trainer
            elif 'trainer' in self.trainer_objects:
                trainer = self.trainer_objects['trainer']
        
        step = state.global_step if hasattr(state, 'global_step') else self.current_step
        self.current_step = step

        self._sync_trainer_objects(trainer)
        
        if hasattr(trainer, 'optimizer') and trainer.optimizer and hasattr(trainer.optimizer, 'param_groups'):
            self.history['learning_rates'].append((step, trainer.optimizer.param_groups[0]['lr']))
        
        model = model or (trainer.model if hasattr(trainer, 'model') else self.model)
        if self.reference_batch is not None and model is not None:
            eval_steps = getattr(args, 'eval_steps', None)
            if step % eval_steps == 0:
                ref_losses_list = self._compute_reference_loss(trainer, model)
                
                if ref_losses_list is not None:
                    avg_ref_loss = sum(ref_losses_list) / len(ref_losses_list) if ref_losses_list else None
                    
                    if avg_ref_loss is not None:
                        if 'ref_loss' not in self.history:
                            self.history['ref_loss'] = []
                        self.history['ref_loss'].append((step, avg_ref_loss))
                        logging.info(f"Computed reference_loss at step {step}: average loss = {avg_ref_loss:.6f}")

        return control
        
    def on_init_end(self, args, state, control, **kwargs):
        """Called after Trainer initialization - setup optimizer and scheduler via hooks"""
        trainer = kwargs.get('trainer')
        if not trainer and self.base_trainer and hasattr(self.base_trainer, '_trainer'):
            trainer = self.base_trainer._trainer

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
            'save_steps': 500,
            'eval_steps': 500,
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
    
    def __init__(self, params: Optional[Dict] = None, model=None, **kwargs):
        super().__init__(model=model, params=params)
        if 'ref_loss' not in self.history:
            self.history['ref_loss'] = []
        if self.model is None and params is not None:
            self._resolve_model() 
            
        self.tokenizer = self.params.get("tokenizer", None)
        
        self.batch_type = self.learning_params.get('batch_type', None)
        self.calib_batch_limit = self.learning_params.get('calib_batch_limit', None)
        self.num_reference_batches = self.params.get('num_reference_batches', 100)

        
        self.hooks = HooksCollection()
        
        self.default_training_args = self.DEFAULT_TRAINING_ARGS.copy()
        if params:
            training_params = {k: v for k, v in params.items() 
                             if k not in ['model', 'tokenizer', 'custom_learning_params']}
            self.default_training_args.update(training_params)
        
        self.gradient_accumulation_steps = self.params.get("gradient_accumulation_steps", None)
        if self.gradient_accumulation_steps is None:
            self.gradient_accumulation_steps = self.default_training_args.get("gradient_accumulation_steps", 1)
        self.align_ref_batches_to_grad_accum = self.params.get("align_ref_batches_to_grad_accum", True)

        self.default_generation_params = self.DEFAULT_GENERATION_PARAMS.copy()
        if params:
            generation_params = {
                k: v for k, v in params.items() 
                if k in self.DEFAULT_GENERATION_PARAMS
            }
            self.default_generation_params.update(generation_params)
        
        self._trainer = None
        self._training_args = None
        self._data_collator = None
        self._fedcore_callback = None
        self._hooks_initialized = False
        self.task_type = None

        self._partial_state = None
        self._accelerator = None
    
    def _init_hooks(self) -> None:
        """Initialize FedCore hooks based on parameters"""
        if self._hooks_initialized:
            return
                
        base_hooks = list(chain(LOGGING_HOOKS, MODEL_LEARNING_HOOKS))
        
        for hook_type in base_hooks:
            if not hook_type.check_init(self.params):
                continue
            hook_instance = hook_type(self)
            self.hooks.append(hook_instance)
        
        for hook_type in self._additional_hooks:
            if not hook_type.check_init(self.params):
                continue
            hook_instance = hook_type(self)
            self.hooks.append(hook_instance)
        
        self._hooks_initialized = True
    
    def execute_hooks(self, hook_type: str, epoch: int, **kwargs) -> None:
        """Execute hooks using HooksCollection instead of dict-based hooks_collection"""
        if hook_type == 'start':
            hooks_list = self.hooks.start()
        elif hook_type == 'end':
            hooks_list = self.hooks.end()
        else:
            hooks_list = []
        
        for hook in hooks_list:
            hook(epoch=epoch, trainer_objects=self.trainer_objects, 
                 history=self.history, **kwargs)
    
    def save_model(self, path: str) -> None:
        """Save model using transformers approach if available.
        
        Uses Transformers Trainer.save_model() if available, otherwise falls back
        to base implementation.
        
        Args:
            path: Path where to save the model
        """
        import logging
        logging.info(f"Saving LLM model to {path}...")
        
        if self.model is None:
            raise ValueError("Model is not initialized. Call fit() first or provide model during initialization.")
        
        if self._trainer is not None and hasattr(self._trainer, 'save_model'):
            self._trainer.save_model(path)
        elif hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(path)
        else:
            super().save_model(path)
    
    def load_model(self, path: str) -> None:
        """Load model using transformers approach if available.
        
        Args:
            path: Path to load the model from
        """        
        if self.model is None:
            raise ValueError("Model is not initialized. Cannot load without model architecture.")
        super().load_model(path)
    
    @property
    def epochs(self) -> int:
        """Get the number of training epochs.
        
        Priority: TrainingArguments > params > default_training_args
        """
        if self._training_args is not None:
            return int(self._training_args.num_train_epochs)
        epochs = self.params.get("epochs") or self.default_training_args.get('num_train_epochs', 3)
        return int(epochs)
    
    def _resolve_model(self, input_data: Optional[CompressionInputData] = None) -> None:
        """Centralized model resolution from various sources"""
        if self.model is not None:
            return
        
        if self._trainer is not None and self._trainer.model is not None:
            self.model = self._trainer.model
            return
        
        if input_data is not None:
            candidate_model = getattr(input_data, 'model', None)
            if candidate_model is None and hasattr(input_data, 'target'):
                candidate_model = input_data.target
            if candidate_model is not None:
                self.model = candidate_model
                return
        
        if self.params:
            model = self.params.get("model", None)
            if model is None and isinstance(self.params.get('custom_learning_params'), dict):
                model = self.params['custom_learning_params'].get('model')
            if model is not None:
                self.model = model
                return
        
        raise RuntimeError(
            "Model is not available. "
            "Call fit() first or provide model during initialization."
        )
        
    def _prepare_data(self, input_data: Union[InputData, CompressionInputData]) -> Dict[str, Dataset]:
        """Convert InputData/CompressionInputData to transformers Dataset format"""
        train_data = self._dataloader_to_dataset(input_data.train_dataloader)
        eval_data = self._dataloader_to_dataset(input_data.val_dataloader)
            
        return {
            'train_dataset': train_data,
            'eval_dataset': eval_data
        }
    
    def prepare_reference_batch(self, input_data: CompressionInputData, use_test_dataloader: bool = False, num_reference_batches: int = None):
        """
            Prepare and save the reference batch.

            Args:
            input_data: CompressionInputData with dataloaders
            use_test_dataloader: If True, use test_dataloader; otherwise, use val_dataloader

            Returns:
            dict: The saved reference batch or None if saving failed
        """
        if self._fedcore_callback is None:
            self._fedcore_callback = FedCoreTransformersTrainer(
                model=self.model,
                base_trainer=self
            )
        
        if self._fedcore_callback._reference_batch_saved:
            logging.info("Reference batch already saved, skipping...")
            return self._fedcore_callback.reference_batch
        
        if num_reference_batches is None:
            num_reference_batches = getattr(self, "num_reference_batches", 100)

        if getattr(self, "align_ref_batches_to_grad_accum", True):
            grad_accum = getattr(self, "gradient_accumulation_steps", 1)
            if grad_accum > 1 and num_reference_batches > 1:
                original_num = num_reference_batches
                num_reference_batches = ((num_reference_batches + grad_accum - 1) // grad_accum) * grad_accum
                if num_reference_batches != original_num:
                    logging.info(
                        f"Adjusted num_reference_batches from {original_num} to {num_reference_batches} "
                        f"(multiple of gradient_accumulation_steps={grad_accum})"
                    )

        dataloader = None
        if use_test_dataloader:
            dataloader = getattr(input_data, 'test_dataloader', None)
            if dataloader is None:
                logging.warning("test_dataloader not available, falling back to val_dataloader")
                dataloader = getattr(input_data, 'val_dataloader', None)
        else:
            dataloader = getattr(input_data, 'val_dataloader', None)
            if dataloader is None:
                logging.warning("val_dataloader not available, falling back to test_dataloader")
                dataloader = getattr(input_data, 'test_dataloader', None)
        
        def _batch_to_ref(batch):
            if batch is None:
                return None
            if isinstance(batch, tuple):
                ref = {
                    "input_ids": batch[0] if len(batch) > 0 else None,
                    "labels": batch[1] if len(batch) > 1 else None,
                    "attention_mask": batch[2] if len(batch) > 2 else None,
                }
                ref = {k: v for k, v in ref.items() if v is not None}
            elif isinstance(batch, dict):
                ref = self._normalize_kwargs(batch, {"input_ids", "attention_mask", "labels"})
                ref = {k: v for k, v in ref.items() if v is not None}
            else:
                return None
            if ref.get("input_ids") is None:
                return None
            return ref

        try:
            n = max(1, num_reference_batches)
            ref_batches = []
            for i, batch in enumerate(dataloader):
                if i >= n:
                    break
                ref = _batch_to_ref(batch)
                if ref is not None:
                    ref_batches.append(copy.deepcopy(ref))
            if not ref_batches:
                logging.error("No valid reference batches collected")
                return None
            self._fedcore_callback.reference_batch = ref_batches
            self._fedcore_callback._reference_batch_saved = True
            total_samples = sum(r["input_ids"].shape[0] for r in ref_batches)
            logging.info(
                f"Reference batches saved successfully before fit(). "
                f"Batches: {len(ref_batches)}, total samples: {total_samples}, "
                f"Source: {'test_dataloader' if use_test_dataloader else 'val_dataloader'}"
            )
            return self._fedcore_callback.reference_batch
        except Exception as e:
            logging.warning(f"Failed to save reference batch before fit(): {e}")
            import traceback
            traceback.print_exc()
            return None


    def _dataloader_to_dataset(self, dataloader) -> Dataset:
        """Convert DataLoader to HuggingFace Dataset"""
        if dataloader is None:
            return None
        
        data = []
        for batch in dataloader:
            if isinstance(batch, dict):
                input_ids = batch.get('input_ids')
                labels = batch.get('labels')
                if labels is None:
                    labels = batch.get('targets')
                attention_mask = batch.get('attention_mask')
            elif isinstance(batch, (list, tuple)):
                input_ids = batch[0] if len(batch) >= 1 else None
                labels = batch[1] if len(batch) >= 2 else None
                attention_mask = batch[2] if len(batch) >= 3 else None
            else:
                continue
            
            if input_ids is None:
                continue
            
            batch_size = input_ids.shape[0]
            for i in range(batch_size):
                data.append({
                    'input_ids': input_ids[i].cpu().tolist(),
                    'labels': None if labels is None else labels[i].cpu().tolist(),
                    'attention_mask': None if attention_mask is None else attention_mask[i].cpu().tolist(),
                })
        
        return Dataset.from_list(data) if data else None

    def _create_trainer(self, datasets: Dict[str, Dataset], for_prediction: bool = False):
        """Create transformers trainer with FedCore integration"""
        self._resolve_model()

        if for_prediction:
            # ---------- PREDICTION MODE: COMPLETELY DISABLE ACCELERATE ----------
            import os
            # Disable Accelerate entirely
            os.environ['ACCELERATE_DISABLE'] = '1'
            os.environ['USE_CPU'] = '1'  # Force CPU mode to avoid CUDA state issues
            
            eval_dataset = datasets.get('eval_dataset')
            if eval_dataset is None:
                raise ValueError("No eval dataset available for prediction")
            
            if hasattr(self, '_trained_model') and self._trained_model is not None:
                model = self._trained_model
            else:
                model = self.model
            
            # Force model to CPU to avoid any CUDA/Accelerate issues
            model = model.cpu()
            model.eval()
            
            eval_batch_size = 8
            if hasattr(self, 'default_training_args'):
                eval_batch_size = self.default_training_args.get('per_device_eval_batch_size', 8)
            
            # CRITICAL: Set no_cuda=True to force CPU mode
            pred_args = TrainingArguments(
                output_dir="./temp_pred_output",
                per_device_eval_batch_size=eval_batch_size,
                local_rank=-1,
                deepspeed=None,
                fp16=False,
                bf16=False,
                remove_unused_columns=False,
                report_to="none",
                save_strategy="no",
                load_best_model_at_end=False,
                disable_tqdm=True,
                no_cuda=True,  # Force CPU, avoid CUDA/Accelerate
                use_cpu=True,  # Additional flag to force CPU
            )
            
            # Create trainer WITHOUT any accelerator
            self._trainer = Trainer(
                model=model,
                args=pred_args,
                eval_dataset=eval_dataset,
                data_collator=getattr(self, '_data_collator', None),
                tokenizer=getattr(self, 'tokenizer', None),
                compute_metrics=getattr(self, 'compute_metrics', None),
                callbacks=None,
            )
            
            # Manually remove accelerator if it somehow got created
            if hasattr(self._trainer, 'accelerator'):
                self._trainer.accelerator = None
            
        else:
            # ---------- TRAINING MODE: Full featured trainer ----------
            # [Keep your existing training code]
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
            
            if self._fedcore_callback is None:
                self._fedcore_callback = FedCoreTransformersTrainer(
                    model=self.model,
                    base_trainer=self
                )
            else:
                self._fedcore_callback.model = self.model
            
            callbacks = [self._fedcore_callback, EarlyStoppingCallback(early_stopping_patience=2000)]
            
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

        saved_reference_batch = None
        saved_reference_batch_flag = False
        if self._fedcore_callback is not None:
            saved_reference_batch = getattr(self._fedcore_callback, 'reference_batch', None)
            saved_reference_batch_flag = getattr(self._fedcore_callback, '_reference_batch_saved', False)
        # self._reset_accelerator_state()

        self._resolve_model(input_data)
        datasets = self._prepare_data(input_data)
        if self._fedcore_callback is None:
            self._fedcore_callback = FedCoreTransformersTrainer(
                model=self.model,
                base_trainer=self
            )
        
        self.prepare_reference_batch(input_data, use_test_dataloader=False, num_reference_batches=self.num_reference_batches)

        
        self._create_trainer(datasets, for_prediction=False)
        if saved_reference_batch is not None and self._fedcore_callback is not None:
            self._fedcore_callback.reference_batch = saved_reference_batch
            self._fedcore_callback._reference_batch_saved = saved_reference_batch_flag
        train_result = self._trainer.train()
        self._trainer.accelerator.wait_for_everyone()
        logging.info(f"Training completed. Loss: {getattr(train_result, 'training_loss', None)}")
        if self._trainer.model is not None:
            self.model = self._trainer.model
            self._trained_model = self.model
        
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
        """Generate long text sequences for input prompts."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not set. Cannot generate text.")
        
        self._resolve_model(input_data)
        datasets = self._prepare_data(input_data)
        if self._trainer is None:
            self._create_trainer(datasets)
        if self._trainer is not None and self._trainer.model is not None:
            self.model = self._trainer.model
        
        model = self.model
        model.eval()
        
        gen_params = self._prepare_generation_params(
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            num_beams=num_beams,
            early_stopping=early_stopping,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **generation_kwargs
        )
        
        eval_dataset = datasets.get('eval_dataset')
        if eval_dataset is None:
            dataloader = (getattr(input_data, 'val_dataloader', None) or 
                        getattr(input_data, 'test_dataloader', None))
            if dataloader:
                eval_dataset = self._dataloader_to_dataset(dataloader)
            else:
                raise ValueError("No dataloader available for generation")
        
        dataloader = self._trainer.get_eval_dataloader(eval_dataset)
        
        generated_texts = []
        
        pbar = tqdm(dataloader, desc="Generating text")
        with torch.no_grad():
            for batch in pbar:
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(model.device)
                
                generate_kwargs = {'input_ids': input_ids, **gen_params}
                if attention_mask is not None:
                    generate_kwargs['attention_mask'] = attention_mask
                
                generated_ids = model.generate(**generate_kwargs)
                
                generated_only = generated_ids[:, input_ids.shape[1]:]
                
                batch_texts = self._decode_token_ids_to_text(generated_only)
                generated_texts.extend(batch_texts)
        
        return generated_texts

    def _prepare_generation_params(self, **kwargs) -> dict:
        """Prepare generation parameters with defaults and tokenizer settings"""
        gen_params = self.default_generation_params.copy()
        gen_params.update({k: v for k, v in kwargs.items() if v is not None})
        
        if self.tokenizer:
            if 'pad_token_id' not in gen_params:
                gen_params['pad_token_id'] = (
                    getattr(self.tokenizer, 'pad_token_id', None) or 
                    getattr(self.tokenizer, 'eos_token_id', None)
                )
            if 'eos_token_id' not in gen_params:
                gen_params['eos_token_id'] = getattr(self.tokenizer, 'eos_token_id', None)
        
        if gen_params.get('do_sample') is False:
            gen_params.pop('temperature', None)
        
        if gen_params.get('max_new_tokens') is not None:
            gen_params.pop('max_length', None)
        
        return gen_params
        
    
    def _decode_token_ids_to_text(self, token_ids: torch.Tensor) -> list:
        """Decode token IDs to text strings using tokenizer"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not available. Cannot decode token IDs to text.")
        
        decoded_texts = []
        if token_ids.dim() > 1:
            for batch_idx in range(token_ids.shape[0]):
                batch_token_ids = token_ids[batch_idx].tolist()
                decoded_text = self.tokenizer.decode(batch_token_ids, skip_special_tokens=True)
                decoded_texts.append(decoded_text)
        else:
            token_ids_list = token_ids.tolist()
            decoded_text = self.tokenizer.decode(token_ids_list, skip_special_tokens=True)
            decoded_texts.append(decoded_text)
        
        return decoded_texts
    
    def _convert_logits_to_output(self, logits: torch.Tensor, output_mode: str = "default") -> Union[torch.Tensor, list]:
        """Convert logits to token IDs or text strings based on output_mode"""
        if output_mode == "raw":
            return logits
    
        if output_mode == "probs":
            probs = F.softmax(logits, dim=-1)
            return probs
        
        if output_mode in ("labels", "default"):
            probs = F.softmax(logits, dim=-1)
            token_ids = torch.argmax(probs, dim=-1)
            return token_ids
        
        if output_mode == "texts":
            if self.tokenizer is None:
                raise ValueError("Tokenizer is not available.")
            probs = F.softmax(logits, dim=-1)
            token_ids = torch.argmax(probs, dim=-1)
            return self._decode_token_ids_to_text(token_ids)
        
    
    def _process_trainer_predictions(self, prediction_output, input_data: CompressionInputData, output_mode: str = "default") -> CompressionOutputData:
        """Process predictions from transformers Trainer.predict() output"""
        if hasattr(prediction_output, 'predictions') and hasattr(prediction_output, 'label_ids'):
            pred_values = torch.tensor(prediction_output.predictions)
            pred_values = self._convert_logits_to_output(pred_values, output_mode)
        else:
            return prediction_output
        
        extracted_fields = self._extract_output_fields(input_data)
        
        checkpoint_info = {}
        if hasattr(self, '_register_model_checkpoint'):
            checkpoint_info = self._register_model_checkpoint(
                model=self.model,
                stage='after'
            )
        
        return CompressionOutputData(
            features=getattr(input_data, 'features', None),
            task=input_data.task,
            predict=pred_values,
            num_classes=extracted_fields.get('num_classes'),
            train_dataloader=extracted_fields.get('train_dataloader'),
            val_dataloader=extracted_fields.get('val_dataloader'),
            data_type=DataTypesEnum.table,
            model=self.model,
            supplementary_data=input_data.supplementary_data,
            checkpoint_path=checkpoint_info.get('checkpoint_path'),
            model_id=checkpoint_info.get('model_id'),
            fedcore_id=checkpoint_info.get('fedcore_id'),
        )
    
    @torch.no_grad()
    def _predict_model(
            self, input_data: CompressionInputData, output_mode: str = "default"
    ):
        
        self._reset_accelerator_state(force_reinit=False)
        self._resolve_model(input_data)
        
        datasets = self._prepare_data(input_data)
        eval_dataset = datasets.get('eval_dataset')
        
        if eval_dataset is None:
            dataloader = (getattr(input_data, 'val_dataloader', None) or 
                        getattr(input_data, 'test_dataloader', None))
            if dataloader:
                eval_dataset = self._dataloader_to_dataset(dataloader)
            else:
                raise ValueError("No dataloader available for prediction")
        
        if not hasattr(self, '_trained_model') or self._trained_model is None:
            if self.model is not None:
                self._trained_model = self.model
            else:
                raise ValueError("No trained model found. Call fit() first.")
        
        # Use the clean trained model
        self.model = self._trained_model
        model = self.model
        # self._create_trainer(datasets, for_prediction=True)
        model = model.cpu()
        model.eval()
        
        # Create dataloader manually
        from torch.utils.data import DataLoader
        
        batch_size = self.default_training_args.get('per_device_eval_batch_size', 8)
        
        data_collator = getattr(self, '_data_collator', None)
        if data_collator is None:
            # Default collator
            def default_collator(features):
                import torch
                batch = {}
                for key in features[0].keys():
                    if isinstance(features[0][key], torch.Tensor):
                        batch[key] = torch.stack([f[key] for f in features])
                    else:
                        batch[key] = torch.tensor([f[key] for f in features])
                return batch
            data_collator = default_collator
        
        dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            collate_fn=data_collator,
            num_workers=0,
            pin_memory=False,
            shuffle=False
        )
        all_predictions = []
        all_labels = []
        
        for batch in dataloader:
            # Move batch to CPU (model is on CPU)
            batch = {k: v.cpu() if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            
            # Get predictions
            if hasattr(outputs, 'logits'):
                predictions = outputs.logits
            else:
                predictions = outputs
                
            all_predictions.append(predictions.cpu())
            
            if 'labels' in batch:
                all_labels.append(batch['labels'].cpu())
        
        # Concatenate predictions
        predictions = torch.cat(all_predictions, dim=0)
        labels = torch.cat(all_labels, dim=0) if all_labels else None
        
        # Create prediction output object
        from transformers.trainer_utils import PredictionOutput
        
        prediction_output = PredictionOutput(
            predictions=predictions.numpy(),
            label_ids=labels.numpy() if labels is not None else None,
            metrics={}
        )
        
        return self._process_trainer_predictions(prediction_output, input_data, output_mode)

    def predict(self, input_data: CompressionInputData,  
                output_mode: str = "default", **generation_kwargs) -> Any:
        """Make predictions using InputData/CompressionInputData"""
        if output_mode == "texts":
            generated_texts = self.generate_long_text(input_data, **generation_kwargs)
            
            extracted_fields = self._extract_output_fields(input_data)
            checkpoint_info = {}
            if hasattr(self, '_register_model_checkpoint'):
                checkpoint_info = self._register_model_checkpoint(
                    model=self.model,
                    stage='after'
                )
            
            return CompressionOutputData(
                task=input_data.task,
                predict=generated_texts,
                num_classes=extracted_fields.get('num_classes'),
                train_dataloader=extracted_fields.get('train_dataloader'),
                val_dataloader=extracted_fields.get('val_dataloader'),
                data_type=DataTypesEnum.table,
                model=self.model,
                supplementary_data=input_data.supplementary_data,
                # checkpoint_path=checkpoint_info.get('checkpoint_path'),
                # model_id=checkpoint_info.get('model_id'),
                # fedcore_id=checkpoint_info.get('fedcore_id'),
            )
        else:
            return self._predict_model(input_data, output_mode)

    def predict_for_fit(self, input_data:CompressionInputData,  
                    output_mode: str = "default") -> Any:
        """Make predictions during training"""
        return self._predict_model(input_data, output_mode)

    
    @property
    def scheduler(self) -> Any:
        """Get scheduler from transformers trainer"""
        if self._fedcore_callback and 'scheduler' in self._fedcore_callback.trainer_objects:
            return self._fedcore_callback.trainer_objects['scheduler']
        return None