""" 
FedCore — A framework for neural network compression and automated model optimization.
FedCore provides tools for model compression techniques including pruning, quantization, 
low-rank decomposition, and knowledge distillation. It integrates with Fedot for automated 
machine learning workflows and supports various parameter-efficient fine-tuning (PEFT) 
strategies for large language models and deep neural networks.
""" 

__version__ = "0.0.1"

import sys
import os
import importlib.util

# Get the absolute path to FedCore root
current_file = os.path.abspath(__file__)  # fedcore/__init__.py
fedcore_dir = os.path.dirname(current_file)  # fedcore/
fedcore_root = os.path.dirname(fedcore_dir)  # FedCore/

# print(f"[fedcore] Current file: {current_file}")
# print(f"[fedcore] fedcore_dir: {fedcore_dir}")
# print(f"[fedcore] fedcore_root: {fedcore_root}")

# Path to external directory
external_dir = os.path.join(fedcore_root, 'external')
# print(f"[fedcore] external_dir: {external_dir}")

# Check if external exists
if os.path.exists(external_dir):
    pass
    # print(f"[fedcore] External directory exists!")
    # print(f"[fedcore] Contents: {os.listdir(external_dir)}")
else:
    # print(f"[fedcore] ERROR: External directory not found at {external_dir}")
    # print(f"[fedcore] Current directory contents: {os.listdir('.')}")
    # # Try to find it
    # print(f"[fedcore] Looking for external in parent...")
    parent = os.path.dirname(fedcore_root)
    if os.path.exists(os.path.join(parent, 'external')):
        external_dir = os.path.join(parent, 'external')
        # print(f"[fedcore] Found external at: {external_dir}")

# Load external as a module
external_module = None

if os.path.exists(external_dir):
    # Check for __init__.py
    init_file = os.path.join(external_dir, '__init__.py')
    if os.path.exists(init_file):
        # print(f"[fedcore] Found __init__.py in external")
        
        # Load external as a module
        spec = importlib.util.spec_from_file_location(
            "external",
            init_file,
            submodule_search_locations=[external_dir]
        )
        external_module = importlib.util.module_from_spec(spec)
        sys.modules["external"] = external_module
        spec.loader.exec_module(external_module)
        
        # Also add it as fedcore.external
        sys.modules["fedcore.external"] = external_module
        
        # print(f"[fedcore] Successfully loaded external module")
        # print(f"[fedcore] Available in external: {[x for x in dir(external_module) if not x.startswith('_')]}")
    else:
        # print(f"[fedcore] No __init__.py found in external")
        # Maybe it's a single file or directory without __init__.py
        # Add to sys.path and try to import
        sys.path.insert(0, external_dir)
        try:
            import external
            external_module = external
            # print(f"[fedcore] Imported external via sys.path")
        except ImportError as e:
            # print(f"[fedcore] Failed to import external: {e}")
            pass

# Make external available
if external_module:
    external = external_module
    __all__ = ['external']
    # print(f"[fedcore] external is now available as fedcore.external")
else:
    # print(f"[fedcore] WARNING: Could not load external module")
    # Create a dummy module
    from types import ModuleType
    external = ModuleType('external')
    __all__ = ['external']

    

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