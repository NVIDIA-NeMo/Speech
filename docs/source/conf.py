#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import sys
import glob

import sphinx_book_theme

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../../nemo"))

from package_info import __version__

templates_path = ["_templates"]

autodoc_mock_imports = [
    'torch',
    'torch.nn',
    'torch.utils',
    'torch.optim',
    'torch.utils.data',
    'torch.utils.data.sampler',
    'torchtext',
    'torchvision',
    'ruamel.yaml',  # ruamel.yaml has ., which is troublesome for this regex
    'hydra',  # hydra-core in requirements, hydra during import
    'dateutil',  # part of core python
    'transformers.tokenization_bert',  # has ., troublesome for this regex
    'sklearn',  # scikit_learn in requirements, sklearn in import
    'nemo_text_processing.inverse_text_normalization',  # Not installed automatically
    'nemo_text_processing.text_normalization',  # Not installed automatically
    'attr',  # attrdict in requirements, attr in import
    'torchmetrics',  # inherited from PTL
    'lightning_utilities',  # inherited from PTL
    'lightning_fabric',
    'apex',
    'megatron.core',
    'transformer_engine',
    'joblib',  # inherited from optional code
    'IPython',
    'ipadic',
    'psutil',
    'regex',
]

_skipped_autodoc_mock_imports = ['wrapt', 'numpy']

for req_path in sorted(list(glob.glob("../../requirements/*.txt"))):
    if "docs.txt" in req_path:
        continue

    req_file = os.path.abspath(os.path.expanduser(req_path))
    with open(req_file, 'r') as f:
        for line in f:
            line = line.replace("\n", "")
            req = re.search(r"([a-zA-Z0-9-_]*)", line)
            if req:
                req = req.group(1)
                req = req.replace("-", "_")

                if req not in autodoc_mock_imports:
                    if req in _skipped_autodoc_mock_imports:
                        print(f"Skipping req : `{req}` (lib {line})")
                        continue

                    autodoc_mock_imports.append(req)
                    print(f"Adding req : `{req}` to autodoc mock requirements (lib {line})")
                else:
                    print(f"`{req}` already added to autodoc mock requirements (lib {line})")

#
# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "sphinxcontrib.bibtex",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinxext.opengraph",
]

bibtex_bibfiles = [
    'asr/asr_all.bib',
    'nlp/nlp_all.bib',
    'nlp/text_normalization/tn_itn_all.bib',
    'tools/tools_all.bib',
    'tts/tts_all.bib',
    'text_processing/text_processing_all.bib',
    'core/adapters/adapter_bib.bib',
]

intersphinx_mapping = {
    'pytorch': ('https://pytorch.org/docs/stable', None),
    'pytorch-lightning': ('https://pytorch-lightning.readthedocs.io/en/latest/', None),
}

# Set default flags for all classes.
autodoc_default_options = {'members': None, 'undoc-members': None, 'show-inheritance': True}

locale_dirs = ['locale/']  # path is example but recommended.
gettext_compact = False  # optional.

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "NVIDIA NeMo"
copyright = "© 2021-2023 NVIDIA Corporation & Affiliates. All rights reserved."
author = "NVIDIA CORPORATION"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.


# The short X.Y version.
# version = "0.10.0"
version = __version__
# The full version, including alpha/beta/rc tags.
# release = "0.9.0"
release = __version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "default"

### Previous NeMo theme
# # NVIDIA theme settings.
# html_theme = 'nvidia_theme'

# html_theme_path = ["."]

# html_theme_options = {
#     'display_version': True,
#     'project_version': version,
#     'project_name': project,
#     'logo_path': None,
#     'logo_only': True,
# }
# html_title = 'Introduction'

# html_logo = html_theme_options["logo_path"]

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "nemodoc"

### from TLT conf.py
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_theme = "sphinx_book_theme"
html_logo = os.path.join('nv_logo.png')
html_title = 'NVIDIA NeMo'

html_theme_options = {
    'logo_only': True,
    'display_version': True,
    # 'prev_next_buttons_location': 'bottom',
    # 'style_external_links': False,
    # 'style_nav_header_background': '#000000',
    # Toc options
    'collapse_navigation': False,
    # 'sticky_navigation': False,
    'navigation_depth': 10,
    # 'includehidden': False,
    # 'titles_only': False,
    # Sphinx Book theme,
    'repository_url': 'https://github.com/NVIDIA/NeMo',
    'use_repository_button': True,
    'show_navbar_depth': 1,
    'show_toc_level': 10,
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

html_favicon = 'favicon.ico'

html_static_path = ['_static']

html_last_updated_fmt = ''


def setup(app):
    app.add_css_file('css/custom.css')
    app.add_js_file('js/pk_scripts.js')


# html_css_files = [
#     './custom.css',
# ]

# html_js_files = [
#     './pk_scripts.js',
# ]

# OpenGraph settings
ogp_site_url = 'https://nvidia.github.io/NeMo/'
ogp_image = 'https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/_static/nv_logo.png'

# MathJax CDN
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/mml-chtml.min.js"
