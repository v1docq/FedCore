import torch

from typing import Optional, Dict, Any, List
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction

from fedcore.models.network_impl.utils.hooks_collection import HooksCollection
from fedcore.models.network_impl.utils.hooks import BaseHook, LoggingHooks, ModelLearningHooks


class FedCoreTransformersTrainer(Trainer):
    """Transformers Trainer with FedCore hooks integration"""
    
    def __init__(
        self,
        model=None,
        args: TrainingArguments = None,
        hooks_collection: Optional[HooksCollection] = None,
        hooks_params: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__(model=model, args=args, **kwargs)
        
        self.hooks_collection = hooks_collection or HooksCollection()
        self.hooks_params = hooks_params or {}
        self.trainer_objects = {
            'optimizer': None,
            'scheduler': None,
            'trainer': self,
            'stop': False
        }
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        if hooks_params:
            self._init_hooks()
    
    def _init_hooks(self):
        hook_enums = [LoggingHooks, ModelLearningHooks]
        
        for hook_enum in hook_enums:
            for hook_elem in hook_enum:
                hook_class = hook_elem.value
                if hook_class.check_init(self.hooks_params):
                    hook_instance = hook_class(self.hooks_params, self.model)
                    self.hooks_collection.append(hook_instance)
    
    def on_epoch_begin(self, args: TrainingArguments, state, control, **kwargs):
        super().on_epoch_begin(args, state, control, **kwargs)
        
        epoch = state.epoch if hasattr(state, 'epoch') else 0
        for hook in self.hooks_collection.start:
            hook(
                epoch=epoch,
                trainer_objects=self.trainer_objects,
                history=self.history,
                **kwargs
            )
        
        if self.trainer_objects.get('stop', False):
            control.should_training_stop = True
    
    def on_epoch_end(self, args: TrainingArguments, state, control, **kwargs):
        """Override epoch end to execute FedCore hooks"""
        super().on_epoch_end(args, state, control, **kwargs)
        
        epoch = state.epoch if hasattr(state, 'epoch') else 0
        if hasattr(state, 'log_history') and state.log_history:
            latest_log = state.log_history[-1]
            if 'loss' in latest_log:
                self.history['train_loss'].append((epoch, latest_log['loss']))
            if 'eval_loss' in latest_log:
                self.history['val_loss'].append((epoch, latest_log['eval_loss']))
        
        for hook in self.hooks_collection.end:
            hook(
                epoch=epoch,
                trainer_objects=self.trainer_objects,
                history=self.history,
                val_loader=self.get_eval_dataloader(self.eval_dataset) if self.eval_dataset else None,
                criterion=self.compute_loss,
                **kwargs
            )
        
        if self.trainer_objects.get('stop', False):
            control.should_training_stop = True
    
    def on_step_end(self, args: TrainingArguments, state, control, **kwargs):
        """Override step end if you have step-level hooks"""
        super().on_step_end(args, state, control, **kwargs)

    
    def training_step(self, model, inputs):
        """Override training step to integrate with trainer objects"""
        if self.trainer_objects['optimizer'] is None and self.optimizer is not None:
            self.trainer_objects['optimizer'] = self.optimizer
        if self.trainer_objects['scheduler'] is None and self.lr_scheduler is not None:
            self.trainer_objects['scheduler'] = self.lr_scheduler
        
        return super().training_step(model, inputs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss, compatible with FedCore hook system"""
        loss = super().compute_loss(model, inputs, return_outputs)
        return loss
    
    def create_optimizer(self):
        """Override optimizer creation to use FedCore optimizer hooks"""
        if self.trainer_objects.get('optimizer') is not None:
            self.optimizer = self.trainer_objects['optimizer']
            return self.optimizer
        
        return super().create_optimizer()
    
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """Override scheduler creation to use FedCore scheduler hooks"""
        if self.trainer_objects.get('scheduler') is not None:
            self.lr_scheduler = self.trainer_objects['scheduler']
            return self.lr_scheduler
        
        return super().create_scheduler(num_training_steps, optimizer)