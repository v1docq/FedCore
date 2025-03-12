import logging
from typing import Union, Callable, List

from fedcore.interfaces.fedcore_optimizer import FedcoreEvoOptimizer
from fedcore.repository.constanst_repository import FEDOT_ASSUMPTIONS
from fedcore.repository.model_repository import default_fedcore_availiable_operation


class ConfigTemplate:
    def __init__(self):
        self.keys = {}
        self.config = {}

    def build(self, config: dict = None):
        for key, method in self.keys.items():
            val = method(config[key]) if key in config.keys() else method()
            self.config.update({key: val})
        return self


class ApiManager(ConfigTemplate):
    def __init__(self, **kwargs):
        super().__init__()
        self.null_state_object()
        self.logger = logging.getLogger("FedCoreAPI")
        self.keys = {'device_config': self.with_device_config,
                     'automl_config': self.with_automl_config,
                     'learning_config': self.with_learning_config,
                     'compute_config': self.with_compute_config}
        self.optimisation_agent = {"Fedcore": FedcoreEvoOptimizer}

    def null_state_object(self):
        self.solver = None
        self.predicted_probs = None
        self.original_model = None

    def with_device_config(self, config: dict):
        self.device_config = DeviceConfig().build(config)
        return self.device_config

    def with_automl_config(self, config: dict):
        self.automl_config = AutomlConfig().build(config)
        return self.automl_config

    def with_learning_config(self, config: dict):
        self.learning_config = LearningConfig().build(config)
        return self.learning_config

    def with_compute_config(self, config: dict):
        self.compute_config = ComputationalConfig().build(config)
        return self.compute_config

    def build(self, config: dict = None):
        for key, method in self.keys.items():
            if key in config.keys():
                method(config[key])
            else:
                method()
        available_operation = default_fedcore_availiable_operation(self.learning_config.peft_strategy)
        if self.automl_config.config['available_operations'] is None:
            self.automl_config.config.update({'available_operations': available_operation})

        return self


class DeviceConfig(ConfigTemplate):
    def __init__(self):
        super().__init__()
        self.keys = {'device': self.with_device,
                     'inference': self.with_inference}

    def with_device(self, task: str = None):
        self.task = task
        return self.task

    def with_inference(self, learning_strategy_params: dict = None):
        self.learning_strategy_params = learning_strategy_params
        return self.learning_strategy_params


class ComputationalConfig(ConfigTemplate):
    def __init__(self):
        super().__init__()
        self.keys = {'backend': self.with_backend,
                     'distributed': self.with_distributed,
                     'output_folder': self.with_output_folder,
                     'use_cache': self.with_cache,
                     'automl_folder': self.with_automl_folder}

    def with_backend(self, backend: str = None):
        self.backend = backend
        return self.backend

    def with_distributed(self, distributed: dict = None):
        self.distributed = distributed
        return self.distributed

    def with_output_folder(self, peft_strategy: dict = None):
        self.peft_strategy = peft_strategy
        return self.peft_strategy

    def with_cache(self, cache_dict: dict = None):
        self.cache = cache_dict
        return self.cache

    def with_automl_folder(self, automl_folder: dict = None):
        self.automl_folder = automl_folder
        return self.automl_folder


class AutomlConfig(ConfigTemplate):
    def __init__(self):
        super().__init__()
        self.keys = {'problem': self.with_problem,
                     'initial_assumption': self.with_initial_assumption,
                     'task_params': self.with_task_params,
                     'timeout': self.with_timeout,
                     'pop_size': self.with_pop_size,
                     'available_operations': self.with_available_operations,
                     'optimizer': self.with_optimizer}

    def with_problem(self, problem: str = None):
        self.problem = problem
        return self.problem

    def with_task_params(self, task_params: dict = None):
        self.task_params = task_params
        return self.task_params

    def with_initial_assumption(self, initial_assumption: str = None):
        self.initial_assumption = initial_assumption
        return self.initial_assumption

    def with_timeout(self, timeout: int = 10):
        self.timeout = timeout
        return self.timeout

    def with_pop_size(self, pop_size: int = 1):
        self.pop_size = pop_size
        return self.pop_size

    def with_available_operations(self, available_operations: List[str] = None):
        self.available_operations = available_operations
        return self.available_operations

    def with_optimizer(self, optimisation_strategy: dict = None):
        self.optimizer = optimisation_strategy
        return self.optimizer


class LearningConfig(ConfigTemplate):
    def __init__(self):
        super().__init__()
        self.keys = {'learning_strategy': self.with_learning_strategy,
                     'learning_strategy_params': self.with_learning_strategy_params,
                     'peft_strategy': self.with_peft_strategy,
                     'peft_strategy_params': self.with_peft_strategy_params,
                     'loss': self.with_loss}

    def with_learning_strategy(self, learning_strategy: str = None):
        self.learning_strategy = learning_strategy
        return self.learning_strategy

    def with_learning_strategy_params(self, learning_strategy_params: dict = None):
        self.learning_strategy_params = learning_strategy_params
        return self.learning_strategy_params

    def with_peft_strategy_params(self, peft_strategy_params: dict = None):
        self.peft_strategy_params = peft_strategy_params
        return self.peft_strategy_params

    def with_peft_strategy(self, peft_strategy: dict = None):
        self.peft_strategy = peft_strategy
        return self.peft_strategy

    def with_loss(self, loss: Union[Callable, str, dict] = None):
        self.loss = loss
        return self.loss
