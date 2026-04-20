# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import argparse
import importlib
import inspect
import os
import traceback

import torch
import wrapt

from nemo.core import Model
from nemo.utils import model_utils
from nemo.utils.import_utils import UnavailableError

DOMAINS = ['asr', 'tts', 'lightning', 'core']


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--domain', choices=DOMAINS, type=str)

    args = parser.parse_args()
    return args


###############################


def _build_import_path(domain, subdomains: list, imp):
    if domain in ["lightning", "core"]:
        import_path = ["nemo", domain]
        import_path.extend(subdomains)
        import_path.append(imp)
    else:
        import_path = ["nemo", "collections", domain]
        import_path.extend(subdomains)
        import_path.append(imp)

    path = ".".join(import_path)
    return path


def _get_class_from_path(domain, subdomains, imp):
    path = _build_import_path(domain, subdomains, imp)

    class_ = None
    result = None

    try:
        class_ = model_utils.import_class_by_path(path)

        if inspect.isclass(class_):
            # Is class wrpped in a wrapt.decorator a the class level? Unwrap for checks.
            if isinstance(class_, wrapt.FunctionWrapper):
                class_ = class_.__wrapped__

            # Subclass tests
            if issubclass(class_, (Model, torch.nn.Module)):
                result = class_
        else:
            class_ = None

        error = None
    # UnavailableError is raised for safe import modules and should not be considered import errors
    except UnavailableError:
        result = None
        error = None

    except Exception:
        error = traceback.format_exc()

    return class_, result, error


def _test_domain_module_imports(module, domain, subdomains: list):
    module_list = []
    failed_list = []
    error_list = []

    error = None
    if len(subdomains) > 0:
        basepath = module.__path__[0]
        nemo_index = basepath.rfind("nemo")
        basepath = basepath[nemo_index:].replace(os.path.sep, ".")
        new_path = '.'.join([basepath, *subdomains])

        try:
            module = importlib.import_module(new_path)
        except Exception:
            print(f"Could not import `{new_path}` ; Traceback below :")
            error = traceback.format_exc()
            error_list.append(error)

    if error is None:
        for imp in dir(module):
            class_, result, error = _get_class_from_path(domain, subdomains, imp)

            if result is not None:
                module_list.append(class_)

            elif class_ is not None:
                failed_list.append(class_)

            if error is not None:
                error_list.append(error)

    for module in module_list:
        print("Module successfully imported :", module)

    print()
    for module in failed_list:
        print("Module did not match a valid signature of NeMo Model (hence ignored):", module)

    print()
    if len(error_list) > 0:
        print("Imports crashed with following traceback !")

        for error in error_list:
            print("*" * 100)
            print()
            print(error)
            print()
            print("*" * 100)
            print()

    if len(error_list) > 0:
        return False
    else:
        return True


###############################


def test_domain_asr(args):
    import nemo.collections.asr as nemo_asr

    all_passed = _test_domain_module_imports(nemo_asr, domain=args.domain, subdomains=['models'])

    if not all_passed:
        exit(1)


def test_domain_tts(args):
    import nemo.collections.tts as nemo_tts

    all_passed = _test_domain_module_imports(nemo_tts, domain=args.domain, subdomains=['models'])

    if not all_passed:
        exit(1)


def test_domain_lightning(args):
    import nemo.lightning as nemo_lightning

    # Test lightning module imports
    all_passed = _test_domain_module_imports(nemo_lightning, domain=args.domain, subdomains=[])

    if not all_passed:
        exit(1)


def test_domain_core(args):
    import nemo.core as nemo_core

    # Test core module imports
    all_passed = _test_domain_module_imports(nemo_core, domain=args.domain, subdomains=[])

    # Test core submodules
    all_passed = _test_domain_module_imports(nemo_core, domain=args.domain, subdomains=["classes"]) and all_passed
    all_passed = _test_domain_module_imports(nemo_core, domain=args.domain, subdomains=["config"]) and all_passed
    all_passed = _test_domain_module_imports(nemo_core, domain=args.domain, subdomains=["connectors"]) and all_passed
    all_passed = _test_domain_module_imports(nemo_core, domain=args.domain, subdomains=["neural_types"]) and all_passed
    all_passed = _test_domain_module_imports(nemo_core, domain=args.domain, subdomains=["optim"]) and all_passed
    all_passed = _test_domain_module_imports(nemo_core, domain=args.domain, subdomains=["utils"]) and all_passed

    if not all_passed:
        exit(1)


###############################


def test_domain(args):
    domain = args.domain

    if domain == 'asr':
        test_domain_asr(args)
    elif domain == 'tts':
        test_domain_tts(args)
    elif domain == 'lightning':
        test_domain_lightning(args)
    elif domain == 'core':
        test_domain_core(args)
    else:
        raise RuntimeError(f"Cannot resolve domain : {domain}")


def run_checks():
    args = process_args()
    test_domain(args)


if __name__ == '__main__':
    run_checks()
