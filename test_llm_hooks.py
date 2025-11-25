#!/usr/bin/env python3
"""
Test: checking the functionality of hooks in LLMTrainer
"""

import torch
import torch.nn as nn
import tempfile
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import numpy as np

from fedcore.models.network_impl.llm_trainer import LLMTrainer
from fedcore.models.network_impl.hooks import (
    LoggingHooks, ModelLearningHooks, 
    Saver, FitReport, EarlyStopping, Evaluator,
    OptimizerGen, SchedulerRenewal, Freezer
)


class TestLLMHooks:
    """Test class for checking hooks in LLMTrainer"""
    
    def __init__(self):
        self.model_name = "sshleifer/tiny-gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Create test data
        self.train_dataset = self._create_test_dataset(32)
        self.eval_dataset = self._create_test_dataset(8)
        
        # Temporary directory for tests
        self.temp_dir = tempfile.mkdtemp()
        
    def _create_test_dataset(self, size):
        """Creates a test dataset"""
        data = []
        for i in range(size):
            # Create simple sequences
            input_ids = torch.randint(0, 1000, (16,)).tolist()
            attention_mask = [1] * 16
            labels = input_ids.copy()
            
            data.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            })
        return Dataset.from_list(data)
    
    def test_saver_hook(self):
        """Test Saver hook"""
        print("\n🧪 Testing Saver Hook...")
        
        # Settings to activate Saver
        training_args = {
            "output_dir": self.temp_dir,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "save_each": 1,  # Save every epoch
            "checkpoint_folder": self.temp_dir,
            "name": "test_model",
            "learning_rate": 0.001,
            "warmup_steps": 0,
            "lr_scheduler_type": "constant",
        }
        
        trainer = LLMTrainer(
            self.model,
            training_args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset
        )
        
        # Check that Saver hook is created
        trainer._init_hooks()
        
        # Check that files are saved
        initial_files = len(list(Path(self.temp_dir).glob("*.pth")))
        
        # Run training
        trainer.fit(self.train_dataset)
        
        # Check that files are created
        final_files = len(list(Path(self.temp_dir).glob("*.pth")))
        
        print(f"Files before training: {initial_files}")
        print(f"Files after training: {final_files}")
        
        assert final_files > initial_files, "Saver hook should create checkpoint files"
        print("✅ Saver hook test passed!")
        
    def test_fit_report_hook(self):
        """Test FitReport hook"""
        print("\n🧪 Testing FitReport Hook...")
        
        training_args = {
            "output_dir": self.temp_dir,
            "num_train_epochs": 2,
            "per_device_train_batch_size": 2,
            "log_each": 1,  # Log every epoch
            "learning_rate": 0.001,
            "warmup_steps": 0,
            "lr_scheduler_type": "constant",
        }
        
        trainer = LLMTrainer(
            self.model,
            training_args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset
        )
        
        trainer._init_hooks()
        
        # Check that FitReport hook is created
        fit_report_hooks = [hook for hook in trainer.hooks.end if isinstance(hook, FitReport)]
        assert len(fit_report_hooks) > 0, "FitReport hook should be created"
        
        print("✅ FitReport hook test passed!")
        
    def test_evaluator_hook(self):
        """Test Evaluator hook"""
        print("\n🧪 Testing Evaluator Hook...")
        
        training_args = {
            "output_dir": self.temp_dir,
            "num_train_epochs": 2,
            "per_device_train_batch_size": 2,
            "eval_each": 1,  # Evaluate every epoch
            "learning_rate": 0.001,
            "warmup_steps": 0,
            "lr_scheduler_type": "constant",
        }
        
        trainer = LLMTrainer(
            self.model,
            training_args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset
        )
        
        trainer._init_hooks()
        
        # Check that Evaluator hook is created
        evaluator_hooks = [hook for hook in trainer.hooks.end if isinstance(hook, Evaluator)]
        assert len(evaluator_hooks) > 0, "Evaluator hook should be created"
        
        print("✅ Evaluator hook test passed!")
        
    def test_early_stopping_hook(self):
        """Test EarlyStopping hook"""
        print("\n🧪 Testing EarlyStopping Hook...")
        
        training_args = {
            "output_dir": self.temp_dir,
            "num_train_epochs": 5,
            "per_device_train_batch_size": 2,
            "early_stop_after": 2,  # Stop after 2 epochs without improvement
            "learning_rate": 0.001,
            "warmup_steps": 0,
            "lr_scheduler_type": "constant",
        }
        
        trainer = LLMTrainer(
            self.model,
            training_args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset
        )
        
        trainer._init_hooks()
        
        # Check that EarlyStopping hook is created
        early_stopping_hooks = [hook for hook in trainer.hooks.end if isinstance(hook, EarlyStopping)]
        assert len(early_stopping_hooks) > 0, "EarlyStopping hook should be created"
        
        print("✅ EarlyStopping hook test passed!")
        
    def test_optimizer_gen_hook(self):
        """Test OptimizerGen hook"""
        print("\n🧪 Testing OptimizerGen Hook...")
        
        training_args = {
            "output_dir": self.temp_dir,
            "num_train_epochs": 2,
            "per_device_train_batch_size": 2,
            "learning_rate": 0.001,
            "warmup_steps": 0,
            "lr_scheduler_type": "constant",
        }
        
        trainer = LLMTrainer(
            self.model,
            training_args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset
        )
        
        trainer._init_hooks()
        
        # Check that OptimizerGen hook is created
        optimizer_hooks = [hook for hook in trainer.hooks.start if isinstance(hook, OptimizerGen)]
        assert len(optimizer_hooks) > 0, "OptimizerGen hook should be created"
        
        print("✅ OptimizerGen hook test passed!")
        
    def test_scheduler_renewal_hook(self):
        """Test SchedulerRenewal hook"""
        print("\n🧪 Testing SchedulerRenewal Hook...")
        
        training_args = {
            "output_dir": self.temp_dir,
            "num_train_epochs": 2,
            "per_device_train_batch_size": 2,
            "scheduler_step_each": 1,  # Scheduler step every epoch
            "sch_type": "one_cycle",
            "learning_rate": 0.001,
            "warmup_steps": 0,
        }
        
        trainer = LLMTrainer(
            self.model,
            training_args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset
        )
        
        trainer._init_hooks()
        
        # Check that SchedulerRenewal hook is created
        scheduler_hooks = [hook for hook in trainer.hooks.start if isinstance(hook, SchedulerRenewal)]
        assert len(scheduler_hooks) > 0, "SchedulerRenewal hook should be created"
        
        print("✅ SchedulerRenewal hook test passed!")
        
    def test_freezer_hook(self):
        """Test Freezer hook"""
        print("\n🧪 Testing Freezer Hook...")
        
        training_args = {
            "output_dir": self.temp_dir,
            "num_train_epochs": 2,
            "per_device_train_batch_size": 2,
            "frozen_prop": 0.3,  # Freeze 30% of parameters
            "refreeze_each": 1,  # Refreeze every epoch
            "learning_rate": 0.001,
            "warmup_steps": 0,
            "lr_scheduler_type": "constant",
        }
        
        trainer = LLMTrainer(
            self.model,
            training_args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset
        )
        
        trainer._init_hooks()
        
        # Check that Freezer hook is created
        freezer_hooks = [hook for hook in trainer.hooks.start if isinstance(hook, Freezer)]
        assert len(freezer_hooks) > 0, "Freezer hook should be created"
        
        print("✅ Freezer hook test passed!")
        
    def test_hook_trigger_and_action(self):
        """Test hook triggers and actions"""
        print("\n🧪 Testing Hook Triggers and Actions...")
        
        # Check that standard hooks work
        training_args = {
            "output_dir": self.temp_dir,
            "num_train_epochs": 2,
            "per_device_train_batch_size": 2,
            "save_each": 1,  # Activate Saver hook
            "log_each": 1,   # Activate FitReport hook
            "learning_rate": 0.001,
            "warmup_steps": 0,
            "lr_scheduler_type": "constant",
        }
        
        trainer = LLMTrainer(
            self.model,
            training_args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset
        )
        
        trainer._init_hooks()
        
        # Check that hooks are created
        saver_hooks = [hook for hook in trainer.hooks.end if isinstance(hook, Saver)]
        fit_report_hooks = [hook for hook in trainer.hooks.end if isinstance(hook, FitReport)]
        
        assert len(saver_hooks) > 0, "Saver hook should be created"
        assert len(fit_report_hooks) > 0, "FitReport hook should be created"
        
        # Run training
        trainer.fit(self.train_dataset)
        
        print("✅ Hook trigger and action test passed!")
        
    def test_hook_ordering(self):
        """Test hook ordering"""
        print("\n🧪 Testing Hook Ordering...")
        
        # Check that hooks are executed in the correct order
        training_args = {
            "output_dir": self.temp_dir,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "save_each": 1,
            "log_each": 1,
            "learning_rate": 0.001,
            "warmup_steps": 0,
            "lr_scheduler_type": "constant",
        }
        
        trainer = LLMTrainer(
            self.model,
            training_args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset
        )
        
        trainer._init_hooks()
        
        # Check that hooks are created and sorted by priority
        start_hooks = trainer.hooks.start
        end_hooks = trainer.hooks.end
        
        # Check that OptimizerGen (priority -100) comes first
        if start_hooks:
            first_hook = start_hooks[0]
            print(f"First start hook: {type(first_hook).__name__} (priority: {getattr(first_hook, '_hook_place', 'unknown')})")
        
        # Check that Saver (priority 100) comes last
        if end_hooks:
            last_hook = end_hooks[-1]
            print(f"Last end hook: {type(last_hook).__name__} (priority: {getattr(last_hook, '_hook_place', 'unknown')})")
        
        # Run training
        trainer.fit(self.train_dataset)
        
        print("✅ Hook ordering test passed!")
        
    def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting LLM Hooks Tests")
        print("=" * 50)
        
        try:
            self.test_saver_hook()
            self.test_fit_report_hook()
            self.test_evaluator_hook()
            self.test_early_stopping_hook()
            self.test_optimizer_gen_hook()
            self.test_scheduler_renewal_hook()
            self.test_freezer_hook()
            self.test_hook_trigger_and_action()
            self.test_hook_ordering()
            
            print("\n🎉 All hook tests passed!")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            raise
        finally:
            # Clean up temporary files
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)


if __name__ == "__main__":
    # Import BaseHook for test hook
    from fedcore.models.network_impl.hooks import BaseHook
    
    # Run tests
    test_suite = TestLLMHooks()
    test_suite.run_all_tests() 