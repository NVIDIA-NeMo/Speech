# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Script to build and install decoder package.

It is used by scripts/asr_language_modeling/ngram_lm/install_beamsearch_decoders.sh to install
KenLM and OpenSeq2Seq decoder.

You can set the order of KenLM model by changing -DKENLM_MAX_ORDER=10 argument.
"""
from __future__ import absolute_import, division, print_function

import argparse
import distutils.ccompiler
import glob
import multiprocessing.pool
import os
import platform
import sys

from setuptools import Extension, distutils, setup

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--num_processes", default=1, type=int, help="Number of cpu processes to build package. (default: %(default)d)"
)
args = parser.parse_known_args()

# reconstruct sys.argv to pass to setup below
sys.argv = [sys.argv[0]] + args[1]


# monkey-patch for parallel compilation
# See: https://stackoverflow.com/a/13176803
def parallelCCompile(
    self,
    sources,
    output_dir=None,
    macros=None,
    include_dirs=None,
    debug=0,
    extra_preargs=None,
    extra_postargs=None,
    depends=None,
):
    # those lines are copied from distutils.ccompiler.CCompiler directly
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
        output_dir, macros, include_dirs, sources, depends, extra_postargs
    )
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

    # parallel code
    def _single_compile(obj):
        try:
            src, ext = build[obj]
        except KeyError:
            return
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

    # convert to list, imap is evaluated on-demand
    thread_pool = multiprocessing.pool.ThreadPool(args[0].num_processes)
    list(thread_pool.imap(_single_compile, objects))
    return objects


def compile_test(header, library):
    dummy_path = os.path.join(os.path.dirname(__file__), "dummy")
    command = (
        "bash -c \"g++ -include "
        + header
        + " -l"
        + library
        + " -x c++ - <<<'int main() {}' -o "
        + dummy_path
        + " >/dev/null 2>/dev/null && rm "
        + dummy_path
        + " 2>/dev/null\""
    )
    return os.system(command) == 0


# hack compile to support parallel compiling
distutils.ccompiler.CCompiler.compile = parallelCCompile

FILES = glob.glob('kenlm/util/*.cc') + glob.glob('kenlm/lm/*.cc') + glob.glob('kenlm/util/double-conversion/*.cc')

FILES += glob.glob('openfst-1.6.3/src/lib/*.cc')

FILES = [fn for fn in FILES if not (fn.endswith('main.cc') or fn.endswith('test.cc') or fn.endswith('unittest.cc'))]

LIBS = ['stdc++']
if platform.system() != 'Darwin':
    LIBS.append('rt')

ARGS = ['-O3', '-DNDEBUG', '-DKENLM_MAX_ORDER=10', '-std=c++11']

if compile_test('zlib.h', 'z'):
    ARGS.append('-DHAVE_ZLIB')
    LIBS.append('z')

if compile_test('bzlib.h', 'bz2'):
    ARGS.append('-DHAVE_BZLIB')
    LIBS.append('bz2')

if compile_test('lzma.h', 'lzma'):
    ARGS.append('-DHAVE_XZLIB')
    LIBS.append('lzma')

os.system('swig -python -c++ ./decoders.i')

decoders_module = [
    Extension(
        name='_swig_decoders',
        sources=FILES + glob.glob('*.cxx') + glob.glob('*.cpp'),
        language='c++',
        include_dirs=['.', 'kenlm', 'openfst-1.6.3/src/include', 'ThreadPool',],
        libraries=LIBS,
        extra_compile_args=ARGS,
    )
]

setup(
    name='ctc_decoders',
    version='1.1',
    description="""CTC decoders""",
    ext_modules=decoders_module,
    py_modules=['ctc_decoders', 'swig_decoders'],
)
