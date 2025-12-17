from typing import Dict, Type, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

from fedcore.api.api_configs import ConfigTemplate, QuestionAnsweringConfigTemplate, SummarizationConfigTemplate

logger = logging.getLogger(__name__)

class NLPPlugin(ABC):
    """Base class for NLP task plugins"""
    
    @abstractmethod
    def get_task_type(self) -> str:
        """Return the task type this plugin handles"""
        pass
    
    @abstractmethod
    def get_config_template(self) -> Type[ConfigTemplate]:
        """Return the configuration template for this task"""
        pass
    
    @abstractmethod
    def create_trainer(self, config: ConfigTemplate, **kwargs):
        """Create appropriate trainer for this NLP task"""
        pass
    
    @abstractmethod
    def get_preprocessing_fn(self, config: ConfigTemplate):
        """Get data preprocessing function for this task"""
        pass


class QuestionAnsweringPlugin(NLPPlugin):
    """Plugin for Question Answering tasks"""
    
    def get_task_type(self) -> str:
        return 'question_answering'
    
    def get_config_template(self) -> Type[ConfigTemplate]:
        return QuestionAnsweringConfigTemplate
    
    def create_trainer(self, config: QuestionAnsweringConfigTemplate, **kwargs):
        from fedcore.models.network_impl.llm_trainer import LLMTrainer
        return LLMTrainer(training_args=config.to_dict(), **kwargs)
    
    def get_preprocessing_fn(self, config: QuestionAnsweringConfigTemplate):
        def qa_preprocessing_fn(examples, tokenizer):
            questions = examples[config.question_column]
            contexts = examples[config.context_column]
            
            tokenized_examples = tokenizer(
                questions,
                contexts,
                truncation=config.truncation,
                max_length=config.max_length,
                stride=config.doc_stride,
                padding=config.padding,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_tensors=config.return_tensors,
            )
            return tokenized_examples
        return qa_preprocessing_fn


class SummarizationPlugin(NLPPlugin):
    """Plugin for Summarization tasks"""
    
    def get_task_type(self) -> str:
        return 'summarization'
    
    def get_config_template(self) -> Type[ConfigTemplate]:
        return SummarizationConfigTemplate
    
    def create_trainer(self, config: SummarizationConfigTemplate, **kwargs):
        from fedcore.models.network_impl.llm_trainer import LLMTrainer
        return LLMTrainer(training_args=config.to_dict(), **kwargs)
    
    def get_preprocessing_fn(self, config: SummarizationConfigTemplate):
        def summarization_preprocessing_fn(examples, tokenizer):
            inputs = examples[config.text_column]
            targets = examples[config.summary_column]
            
            model_inputs = tokenizer(
                inputs,
                max_length=config.max_source_length,
                padding=config.padding,
                truncation=config.truncation,
            )
            
            labels = tokenizer(
                targets,
                max_length=config.max_target_length,
                padding=config.padding,
                truncation=config.truncation,
            )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        return summarization_preprocessing_fn


class NLPPluginManager:
    
    def __init__(self):
        self.plugins: Dict[str, NLPPlugin] = {}
        self._register_builtin_plugins()
    
    def _register_builtin_plugins(self):
        """Register built-in NLP plugins"""
        self.register_plugin(QuestionAnsweringPlugin())
        self.register_plugin(SummarizationPlugin())
    
    def register_plugin(self, plugin: NLPPlugin):
        """Register a new NLP plugin"""
        task_type = plugin.get_task_type()
        self.plugins[task_type] = plugin
        logger.info(f"Registered NLP plugin for task: {task_type}")
    
    def get_plugin(self, task_type: str) -> Optional[NLPPlugin]:
        """Get plugin for specific task type"""
        return self.plugins.get(task_type)
    
    def get_supported_tasks(self) -> list:
        """Get list of supported NLP tasks"""
        return list(self.plugins.keys())
    
    def has_plugin(self, task_type: str) -> bool:
        """Check if plugin exists for task type"""
        return task_type in self.plugins


nlp_plugin_manager = NLPPluginManager()