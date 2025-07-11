import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'CleanEasy'
copyright = '2025, Aman Sonwani'
author = 'Aman Sonwani'
release = '0.2.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon'
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'alabaster'
html_static_path = ['_static']