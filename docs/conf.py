# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath('..'))
project = 'FedCore'
copyright = '2026, FedCore team'
author = 'FedCore team'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",
]

autodoc_mock_imports = [
    'torch',
    'torchvision',
    'numpy',
    'pandas',
    'matplotlib',
    'cv2',
    'tensorflow',
    'fedot',
    'golem',
    'dask',
    'distributed',
    'bokeh',
    'joblib',
    'transformers',
    'accelerate',
    'bitsandbytes',
    'datasets',
    'huggingface_hub',
    'timm',
    'xgboost',
    'onnxruntime',
    'onnxruntime_extensions',
    'tdecomp',
    'loralib',
    'hyperopt',
    'pymonad',
    'fastai',
    'fastcore',
    'neural_insights',
    'segmentation_models_pytorch',
    'tf_slim',
    'pycocotools',
    'torch_pruning',
    'tqdm',
    'scipy',
    'sklearn',
    'PIL',
    'chardet',
    'torchmetrics',
    'chronos',
    'typing_extensions',
    'opendatasets',
    'prettytable',
    'pydantic',
    'schema',
    'statsmodels',
    'imageio',
    'mabwiser',
    'torchinfo',
    'mpi4py',
    'yaml',
    'datasetsforecast',
    'sktime',
    'evaluate',
    'pynvml',
    'seaborn',
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__',
    'inherited-members': False,
}

autodoc_typehints = 'description'
autodoc_duplicate_object_description = 'squash'
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ['_templates']
exclude_patterns = [
    '_build', 'Thumbs.db', '.DS_Store', '.venv',
    'api/modules.rst', 'api/fedcore.rst',
]

suppress_warnings = [
    'autodoc.duplicate_object_description',
    'ref.python',
    'autodoc',
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
