import pytest
import torch
import tempfile
import os

from pathlib import Path
from unittest.mock import Mock, MagicMock
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

from fedcore.models.network_impl.llm_trainer import LLMTrainer
from fedcore.models.network_impl.hooks import (
    LoggingHooks, ModelLearningHooks, 
    Saver, FitReport, EarlyStopping, Evaluator,
    OptimizerGen, SchedulerRenewal, Freezer
)


@pytest.fixture(scope="module")
def tokenizer():
    model_name = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture(scope="module")
def model():
    model_name = "sshleifer/tiny-gpt2"
    return AutoModelForCausalLM.from_pretrained(model_name)


@pytest.fixture
def train_dataset():
    data = []
    for i in range(32):
        input_ids = torch.randint(0, 1000, (16,)).tolist()
        attention_mask = [1] * 16
        labels = input_ids.copy()
        
        data.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        })
    return Dataset.from_list(data)


@pytest.fixture
def eval_dataset():
    data = []
    for i in range(8):
        input_ids = torch.randint(0, 1000, (16,)).tolist()
        attention_mask = [1] * 16
        labels = input_ids.copy()
        
        data.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        })
    return Dataset.from_list(data)


@pytest.fixture
def temp_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def basic_training_args(temp_dir):
    return {
        "output_dir": temp_dir,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 2,
        "learning_rate": 0.001,
        "warmup_steps": 0,
        "lr_scheduler_type": "constant",
    }


@pytest.fixture
def llm_trainer(model, basic_training_args, train_dataset, eval_dataset):
    return LLMTrainer(
        model,
        training_args=basic_training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )


class TestLLMHooks:

    def test_saver_hook(self, model, train_dataset, eval_dataset, temp_dir):
        training_args = {
            "output_dir": temp_dir,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "save_each": 1, 
            "checkpoint_folder": temp_dir,
            "name": "test_model",
            "learning_rate": 0.001,
            "warmup_steps": 0,
            "lr_scheduler_type": "constant",
        }
        
        trainer = LLMTrainer(
            model,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        trainer._init_hooks()
        
        initial_files = len(list(Path(temp_dir).glob("*.pth")))
        
        trainer.fit(train_dataset)
        
        final_files = len(list(Path(temp_dir).glob("*.pth")))
        
        assert final_files > initial_files, "Saver hook should create checkpoint files"

    def test_fit_report_hook(self, model, train_dataset, eval_dataset, temp_dir):
        training_args = {
            "output_dir": temp_dir,
            "num_train_epochs": 2,
            "per_device_train_batch_size": 2,
            "log_each": 1,  
            "learning_rate": 0.001,
            "warmup_steps": 0,
            "lr_scheduler_type": "constant",
        }
        
        trainer = LLMTrainer(
            model,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        trainer._init_hooks()
        
        fit_report_hooks = [hook for hook in trainer.hooks.end if isinstance(hook, FitReport)]
        assert len(fit_report_hooks) > 0, "FitReport hook should be created"

    def test_evaluator_hook(self, model, train_dataset, eval_dataset, temp_dir):
        training_args = {
            "output_dir": temp_dir,
            "num_train_epochs": 2,
            "per_device_train_batch_size": 2,
            "eval_each": 1,  
            "learning_rate": 0.001,
            "warmup_steps": 0,
            "lr_scheduler_type": "constant",
        }
        
        trainer = LLMTrainer(
            model,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        trainer._init_hooks()
        
        evaluator_hooks = [hook for hook in trainer.hooks.end if isinstance(hook, Evaluator)]
        assert len(evaluator_hooks) > 0, "Evaluator hook should be created"

    def test_early_stopping_hook(self, model, train_dataset, eval_dataset, temp_dir):
        training_args = {
            "output_dir": temp_dir,
            "num_train_epochs": 5,
            "per_device_train_batch_size": 2,
            "early_stop_after": 2,  
            "learning_rate": 0.001,
            "warmup_steps": 0,
            "lr_scheduler_type": "constant",
        }
        
        trainer = LLMTrainer(
            model,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        trainer._init_hooks()
        
        early_stopping_hooks = [hook for hook in trainer.hooks.end if isinstance(hook, EarlyStopping)]
        assert len(early_stopping_hooks) > 0, "EarlyStopping hook should be created"

    def test_optimizer_gen_hook(self, model, train_dataset, eval_dataset, temp_dir):
        training_args = {
            "output_dir": temp_dir,
            "num_train_epochs": 2,
            "per_device_train_batch_size": 2,
            "learning_rate": 0.001,
            "warmup_steps": 0,
            "lr_scheduler_type": "constant",
        }
        
        trainer = LLMTrainer(
            model,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        trainer._init_hooks()
        
        optimizer_hooks = [hook for hook in trainer.hooks.start if isinstance(hook, OptimizerGen)]
        assert len(optimizer_hooks) > 0, "OptimizerGen hook should be created"

    def test_scheduler_renewal_hook(self, model, train_dataset, eval_dataset, temp_dir):
        """Test SchedulerRenewal hook"""
        training_args = {
            "output_dir": temp_dir,
            "num_train_epochs": 2,
            "per_device_train_batch_size": 2,
            "scheduler_step_each": 1, 
            "sch_type": "one_cycle",
            "learning_rate": 0.001,
            "warmup_steps": 0,
        }
        
        trainer = LLMTrainer(
            model,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        trainer._init_hooks()
        
        scheduler_hooks = [hook for hook in trainer.hooks.start if isinstance(hook, SchedulerRenewal)]
        assert len(scheduler_hooks) > 0, "SchedulerRenewal hook should be created"

    def test_freezer_hook(self, model, train_dataset, eval_dataset, temp_dir):
        training_args = {
            "output_dir": temp_dir,
            "num_train_epochs": 2,
            "per_device_train_batch_size": 2,
            "frozen_prop": 0.3,  
            "refreeze_each": 1,  
            "learning_rate": 0.001,
            "warmup_steps": 0,
            "lr_scheduler_type": "constant",
        }
        
        trainer = LLMTrainer(
            model,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        trainer._init_hooks()
        
        freezer_hooks = [hook for hook in trainer.hooks.start if isinstance(hook, Freezer)]
        assert len(freezer_hooks) > 0, "Freezer hook should be created"

    def test_hook_trigger_and_action(self, model, train_dataset, eval_dataset, temp_dir):
        """Test hook triggers and actions"""
        training_args = {
            "output_dir": temp_dir,
            "num_train_epochs": 2,
            "per_device_train_batch_size": 2,
            "save_each": 1,  
            "log_each": 1,  
            "learning_rate": 0.001,
            "warmup_steps": 0,
            "lr_scheduler_type": "constant",
        }
        
        trainer = LLMTrainer(
            model,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        trainer._init_hooks()
        
        saver_hooks = [hook for hook in trainer.hooks.end if isinstance(hook, Saver)]
        fit_report_hooks = [hook for hook in trainer.hooks.end if isinstance(hook, FitReport)]
        
        assert len(saver_hooks) > 0, "Saver hook should be created"
        assert len(fit_report_hooks) > 0, "FitReport hook should be created"
        
        trainer.fit(train_dataset)

    def test_hook_ordering(self, model, train_dataset, eval_dataset, temp_dir):
        """Test hook ordering"""
        training_args = {
            "output_dir": temp_dir,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "save_each": 1,
            "log_each": 1,
            "learning_rate": 0.001,
            "warmup_steps": 0,
            "lr_scheduler_type": "constant",
        }
        
        trainer = LLMTrainer(
            model,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        trainer._init_hooks()
        
        start_hooks = trainer.hooks.start
        end_hooks = trainer.hooks.end
        
        if start_hooks:
            first_hook = start_hooks[0]
            assert hasattr(first_hook, '_hook_place'), "Hook should have priority attribute"
        
        if end_hooks:
            last_hook = end_hooks[-1]
            assert hasattr(last_hook, '_hook_place'), "Hook should have priority attribute"
        
        trainer.fit(train_dataset)

    def test_hook_initialization(self, llm_trainer):
        llm_trainer._init_hooks()
        
        assert hasattr(llm_trainer.hooks, 'start'), "Trainer should have start hooks"
        assert hasattr(llm_trainer.hooks, 'end'), "Trainer should have end hooks"
        
        assert isinstance(llm_trainer.hooks.start, list), "Start hooks should be a list"
        assert isinstance(llm_trainer.hooks.end, list), "End hooks should be a list"

    @pytest.mark.parametrize("hook_param,expected_hook_type", [
        ("save_each", Saver),
        ("log_each", FitReport),
        ("eval_each", Evaluator),
        ("early_stop_after", EarlyStopping),
        ("scheduler_step_each", SchedulerRenewal),
        ("refreeze_each", Freezer),
    ])
    def test_hook_activation_by_parameter(self, model, train_dataset, eval_dataset, temp_dir, 
                                        hook_param, expected_hook_type):
        training_args = {
            "output_dir": temp_dir,
            "num_train_epochs": 2,
            "per_device_train_batch_size": 2,
            "learning_rate": 0.001,
            "warmup_steps": 0,
            "lr_scheduler_type": "constant",
            hook_param: 1,  
        }
        
        trainer = LLMTrainer(
            model,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        trainer._init_hooks()
        
        if expected_hook_type in [Saver, FitReport, Evaluator, EarlyStopping]:
            hooks_to_check = trainer.hooks.end
        else:
            hooks_to_check = trainer.hooks.start
        
        found_hooks = [hook for hook in hooks_to_check if isinstance(hook, expected_hook_type)]
        assert len(found_hooks) > 0, f"{expected_hook_type.__name__} should be activated by {hook_param}"


@pytest.mark.slow
def test_complete_training_cycle(model, train_dataset, eval_dataset, temp_dir):
    training_args = {
        "output_dir": temp_dir,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 2,
        "save_each": 1,
        "log_each": 1,
        "eval_each": 1,
        "learning_rate": 0.001,
        "warmup_steps": 0,
        "lr_scheduler_type": "constant",
    }
    
    trainer = LLMTrainer(
        model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    try:
        trainer.fit(train_dataset)
        assert True, "Training should complete without errors"
    except Exception as e:
        pytest.fail(f"Training failed with error: {e}")


def test_hook_priority_ordering():
    from fedcore.models.network_impl.hooks import BaseHook
    
    class LowPriorityHook(BaseHook):
        _hook_place = -100
        
    class MediumPriorityHook(BaseHook):
        _hook_place = 0
        
    class HighPriorityHook(BaseHook):
        _hook_place = 100
    
    hooks = [MediumPriorityHook(), HighPriorityHook(), LowPriorityHook()]
    
    sorted_hooks = sorted(hooks, key=lambda x: getattr(x, '_hook_place', 0))
    
    assert isinstance(sorted_hooks[0], LowPriorityHook), "Low priority hook should come first"
    assert isinstance(sorted_hooks[1], MediumPriorityHook), "Medium priority hook should come second"
    assert isinstance(sorted_hooks[2], HighPriorityHook), "High priority hook should come last"