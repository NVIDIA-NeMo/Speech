# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import re
from dataclasses import dataclass, field

import torch
from lightning.pytorch.callbacks.callback import Callback

from nemo.lightning.io.mixin import IOMixin


def extract_module_attr_name(pl_module: "pl.LightningModule") -> str:
    """Extracts the held nn.Module from a pl.LightningModule, will try "module", "model", or fail.

    Args:
        pl_module (pl.LightningModule): the LightningModule used in training.

    Raises:
        ValueError: if the pl_module has neither a .mdoel or .module

    Returns:
        str: the attr-name of the nn.Module
    """
    if hasattr(pl_module, 'module'):
        return 'module'
    elif hasattr(pl_module, 'model'):
        return 'model'
    else:
        raise ValueError("Expected lightning_module to have a .model or .module attr.")


def listify(x):
    """Wraps input in a list, if not already a list.

    Args:
        x (Anything): the input, can be anything.

    Returns:
        Anything | list(Anything): Anything (if it's already a list) o/w list(Anything)
    """
    if not isinstance(x, list):
        return [x]
    return x


def get_modules_from_selector(model, module_selector):
    """Iterator over model's modules whose FQN match the module_selector.

    Args:
        model (nn.Module): the model to iterate over.
        module_selector (str): module selector, if empty or '*' will return the whole model. If
        there's an asterisk in the name will match it as a regexp.

    Raises:
        AttributeError: if the user provides an invalid selector.
        AttributeError: if user's selector selects a non-nn.Module attribute.

    Yields:
        Iterator(nn.Module): iterator over modules whose FQN matches module_selector
    """
    if module_selector is None or module_selector == '' or module_selector == '*':
        yield model
        return

    assert isinstance(module_selector, str), module_selector
    atoms: List[str] = module_selector.split('.')
    tmp = model

    for i, item in enumerate(atoms):
        if '*' in item:
            # handle wildcard selector
            # TODO(@akoumparouli): support more complex selectors e.g. net_b.*.net_c.*.conv
            for name, module in tmp.named_children():
                if re.match(item.replace('*', '.*'), name):
                    yield module
            return

        if not hasattr(tmp, item):
            raise AttributeError(tmp._get_name() + " has no " "attribute `" + item + "`")
        tmp = getattr(tmp, item)

        if not isinstance(tmp, torch.nn.Module):
            raise AttributeError("`" + item + "` is not " "an nn.Module")

    yield tmp


def compile_module(config, module):
    """Jit-compiles an nn.Module

    Args:
        config (JitConfig): jit config
        module (nn.Module): the module to be compiled

    Returns:
        nn.Module: the (potentially) compiled module
    """
    if config.use_torch:
        module.compile(**config.torch_kwargs)
        return True
    elif config.use_thunder:
        import thunder
        import thunder.dynamo
        from thunder.dev_utils.nvtx_profile_transform import NvtxProfileTransform

        # With this setting, Dynamo Graphs inline all the modules (so Dynamo FXGraph just
        # consists of `call_function` nodes only and no `call_module` node.
        # This is the default setting in PyTorch 2.5 onwards
        # (see https://github.com/pytorch/pytorch/pull/131275)
        torch._dynamo.config.inline_inbuilt_nn_modules = True

        xforms: list = [NvtxProfileTransform()] if config.profile_thunder else []
        module.compile(backend=thunder.dynamo.ThunderCompiler(transforms=xforms))
        return True
    else:
        return False


@dataclass
class JitConfig:
    """Config POD for Jit transforms (e.g. torch.compile or thunder)
    Options:
    - module_selector (str): reg-exp to match modules to apply JitTransform to, useful for multi-trunk
      models where you want to apply it on one of them only. If empty will apply transform to root
      module.
    - use_torch (bool): whether to use torch.compile or not.
    - torch_kwargs (dict): kwargs to pass to torch.compile.
    - use_thunder (bool): whether to use thunder or not.
    - profile_thunder (bool): toggle for thunder's profiler.
    """

    module_selector: str = ''
    use_torch: bool = False
    torch_kwargs: dict = field(default_factory=dict)
    use_thunder: bool = False
    profile_thunder: bool = False

    def __post_init__(self):
        assert not (self.use_torch and self.use_thunder), "use_torch cannot be used at the same time with use_thunder"


class JitTransform(Callback, IOMixin):
    """
    Apply JIT-compling on PyTorch model

    Args:
        config (JitConfig): The jit-compiler config to use.

    Example:
        >>> from nemo.lightning.pytorch.callbacks import JitTransform
        >>> trainer = Trainer(callbacks=[JitTransform(JitConfig(use_torch=True))])
    """

    def __init__(self, config: JitConfig):
        assert config is not None
        self.config = config
        assert not (self.config.use_torch and self.config.use_thunder)

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Jit-compiles the model at the start of the epoch.
        While other events such as on_train_start are more suitable, we use on_train_epoch_start
        since that is what is used in peft (we want to jit after adding the adapters).

        Args:
            trainer (pl.Trainer): PTL trainer
            pl_module (pl.LightningModule): PTL module
        """
        if self.config is None:
            return
        if not self.config.use_thunder and not self.config.use_torch:
            return

        attr_name = extract_module_attr_name(pl_module)
        model = getattr(pl_module, attr_name)

        if getattr(pl_module, '_compiled', False) == True:
            return

        # TODO(@akoumparouli): you want to concatenate (via regex OR-operator) all expressions
        # and trigger the compile if anyone matches, instead of iterating over all O(N^2).
        compiled = False
        for config in listify(self.config):
            for module in get_modules_from_selector(model, config.module_selector):
                compiled |= compile_module(config, module)

        setattr(pl_module, '_compiled', compiled)
