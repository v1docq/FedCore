# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
print(f"Current working directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")

# Добавьте эти пути по очереди для теста
test_paths = [
    os.path.abspath('../../..'),      # fedcore/api
    os.path.abspath('../../../..'),   # fedcore
    os.path.abspath('../../../../..') # FedCore
]

for path in test_paths:
    print(f"\nChecking path: {path}")
    if os.path.exists(os.path.join(path, 'fedcore')):
        print(f"  → Found fedcore package!")
    if os.path.exists(os.path.join(path, 'fedcore', 'api')):
        print(f"  → Found api subpackage!")
    sys.path.insert(0, path)

project = 'FedCore API'
copyright = '2025, Ivan Monnar'
author = 'Ivan Monnar'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode'
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
