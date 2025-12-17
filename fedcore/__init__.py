""" 
FedCore — A framework for neural network compression and automated model optimization.
FedCore provides tools for model compression techniques including pruning, quantization, 
low-rank decomposition, and knowledge distillation. It integrates with Fedot for automated 
machine learning workflows and supports various parameter-efficient fine-tuning (PEFT) 
strategies for large language models and deep neural networks.
""" 

__version__ = "0.0.1"

from fedcore import api
from fedcore import algorithm
from fedcore import architecture
from fedcore import data
from fedcore import inference
from fedcore import interfaces
from fedcore import losses
from fedcore import metrics
from fedcore import models
from fedcore import repository
from fedcore import tools 