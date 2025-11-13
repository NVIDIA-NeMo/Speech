# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import glob
import importlib.machinery
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig
from safetensors import safe_open
from torch import nn

from nemo.utils import logging

# ==============================================================================
# Contants
# ==============================================================================
PYTHON_CONFIG_GETTER_NAME = "get_config"
CHECKPOINT_FORMAT = "checkpoint_{}/ema.safetensors"
CONFIG_NAME = "config.json"
GIT_HASH_NAME = "githash"
SCRIPT_PLACEHOLDER = "[[[<<<SCRIPT_PLACEHOLDER>>>]]]"


# ==============================================================================
# Configuration Class and Utilities
# ==============================================================================
def get_config_from_file(config_path: str) -> DictConfig:
    """
    Loads a configuration from a JSON or Python file.

    - For JSON files (`*.json`), it parses the file directly.
    - For Python files (`*.py`), it imports the file as a module and calls a
      `get_config()` function within it.
    - It also supports a special syntax `path/to/config.py:config_name` to select
      a specific configuration from a Python file that returns a dictionary of configs.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        Config: The loaded configuration object.

    Raises:
        AssertionError: If the file path is invalid, does not exist, or is not in
                        the expected format.
    """

    match = re.search(r".+\.((json)|(py)|(py:.+))$", config_path)
    assert match, f"Only Python (*.py) or JSON (*.json) files are supported, but got {config_path}."

    py_config_name: str | None = None
    if not (config_path.endswith(".py") or config_path.endswith(".json")):
        config_path_split = config_path.split(":")
        config_path = ":".join(config_path_split[:-1])
        py_config_name = config_path_split[-1]

    assert os.path.isfile(config_path), f"Configuration file not found at: {config_path}"

    if config_path.endswith(".json"):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config_module = importlib.machinery.SourceFileLoader("_config", config_path).load_module()
        assert hasattr(
            config_module, PYTHON_CONFIG_GETTER_NAME
        ), f"Python config file must define a `{PYTHON_CONFIG_GETTER_NAME}` function."
        config = getattr(config_module, PYTHON_CONFIG_GETTER_NAME)(py_config_name)
        assert isinstance(config, Mapping), f"`{PYTHON_CONFIG_GETTER_NAME}` must return a dictionary-like object."
    cfg = DictConfig(config)
    return cfg


def get_config() -> DictConfig:
    """
    Parses command-line arguments to load the main configuration for a training run.

    This function implements a hierarchical configuration loading strategy:
    1. It checks if a `config.json` exists in the specified `--workdir`. If so, it loads it.
    2. If a `--config` argument is also provided, it uses that file to update the
       configuration loaded from the work directory.
    3. If no config exists in the work directory, it requires the `--config` argument
       to be provided as the base configuration.

    This allows for resuming training from a work directory while also being able to
    override specific parameters for a new run.

    Returns:
        Config: The final, consolidated configuration object.
    """
    parser = argparse.ArgumentParser(description="Load training configuration.")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to a Python or JSON configuration file.")
    parser.add_argument(
        "-w", "--workdir", type=str, required=True, help="Work directory to save logs and checkpoints."
    )

    args = parser.parse_args()
    workdir_path = args.workdir
    config_save_path = os.path.join(workdir_path, CONFIG_NAME)

    if os.path.exists(config_save_path):
        logging.info(f"Resuming from work directory. Loading configuration from {config_save_path}.")
        cfg = get_config_from_file(config_save_path)
        if args.config and args.config != config_save_path:
            logging.info(f"Updating loaded configuration with parameters from {args.config}.")
            override_cfg = get_config_from_file(args.config)
            cfg.update(override_cfg)
    else:
        assert args.config is not None, "A configuration file must be specified via `-c` or `--config` for a new run."
        logging.info(f"Starting a new run. Loading configuration from {args.config}.")
        cfg = get_config_from_file(args.config)
    cfg.workdir_path = workdir_path

    return cfg


def get_config_from_dir(workdir_path: str) -> DictConfig:
    """
    A simple utility to load the configuration directly from a work directory.

    Args:
        workdir_path (str): The path to the work directory containing a `config.json`.

    Returns:
        DictConfig: The loaded configuration object.
    """
    config_save_path = os.path.join(workdir_path, CONFIG_NAME)
    cfg = get_config_from_file(config_save_path)
    cfg.workdir_path = workdir_path
    return cfg


# ==============================================================================
# Base Model Classes
# ==============================================================================


class PreTrainedModel(nn.Module):
    config_class = DictConfig

    """
    A base class for models to handle loading from pretrained checkpoints.

    This class provides a common interface for initializing a model and loading
    weights from a saved checkpoint, following a pattern similar to libraries
    like Hugging Face's Transformers.

    Args:
        config (DictConfig | dict[str, Any]): A configuration object containing model hyperparameters.
    """

    def __init__(self, config: DictConfig | dict[str, Any], *args, **kwargs):
        super().__init__()
        self.config = config if isinstance(config, self.config_class) else self.config_class(config)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_dir: str,
        cfg: DictConfig | dict[str, Any] | None = None,
        checkpoint_regex: str = "checkpoint_*/ema.safetensors",
        strict: bool = False,
        **model_kwargs,
    ) -> "PreTrainedModel":
        """
        Loads a pretrained model from a directory.

        This method first loads the configuration file from the specified directory,
        initializes the model with this configuration, and then loads the weights
        from the latest checkpoint file found in that directory.

        Args:
            cls (type): The model class to instantiate.
            pretrained_dir (str): The directory containing the pretrained model
                                  config and checkpoint files.
            cfg (DictConfig | dict[str, Any] | None, optional): An optional config object to override
                                           the loaded config. Defaults to None.
            checkpoint_regex (str, optional): A regex pattern to find the checkpoint
                                              file. Defaults to "checkpoint_*/ema.safetensors".
            strict (bool, optional): Whether to strictly enforce that the keys in
                                     the checkpoint match the keys of the model.
                                     Defaults to False.
            **model_kwargs: Additional keyword arguments to pass to the model's
                            constructor.

        Returns:
            PreTrainedModel: An instance of the model with loaded weights.
        """
        pretrained_cfg = get_config_from_dir(pretrained_dir).model
        if cfg is not None:
            pretrained_cfg.update(cfg)
            logging.info(f"The loaded config of the pretrained model is updated to: {pretrained_cfg}")
        model = cls(
            pretrained_cfg,
            **model_kwargs,
        )
        model_state_dict = {}
        with safe_open(latest_checkpoint_path(pretrained_dir, checkpoint_regex), framework="pt", device="cpu") as f:
            for key in f.keys():
                model_state_dict[key] = f.get_tensor(key)
        model.load_state_dict(model_state_dict, strict=strict)
        return model

    def get_optimizer_param_groups(self, weight_decay: float = 0.0) -> list[dict]:
        """
        Separates model parameters into two groups: one with weight decay and one without.

        This is a common practice in training deep learning models, where weight decay
        is typically applied to the weights of linear and convolutional layers, but not
        to biases or normalization layer parameters.

        Args:
            weight_decay (float, optional): The weight decay value to apply to the
                                            first group of parameters. Defaults to 0.0.

        Returns:
            list[dict]: A list of two dictionaries, each suitable for an optimizer's
                        parameter groups. The first group has weight decay, and the
                        second does not.
        """

        def _get_weight_names(module):
            """Recursively finds the names of all 'weight' parameters in conv/linear layers."""
            result = []
            is_weight_layer = isinstance(
                module,
                (
                    nn.Linear
                    | nn.Conv1d
                    | nn.Conv2d
                    | nn.Conv3d
                    | nn.ConvTranspose1d
                    | nn.ConvTranspose2d
                    | nn.ConvTranspose3d
                ),
            )
            if is_weight_layer:
                result.append("weight")
            else:
                for name, child in module.named_children():
                    result += [f"{name}.{n}" for n in _get_weight_names(child)]
            return result

        # Separate parameters
        params_w_decay, params_wo_decay = [], []
        param_names_w_decay = set(_get_weight_names(self))

        for n, p in self.named_parameters():
            if p.requires_grad:
                if n in param_names_w_decay:
                    params_w_decay.append(p)
                else:
                    params_wo_decay.append(p)
        return [
            {"params": params_w_decay, "weight_decay": weight_decay},
            {"params": params_wo_decay, "weight_decay": 0.0},
        ]


# ==============================================================================
# IO and Checkpointing Utilities
# ==============================================================================


def check_git_hash() -> str | None:
    """
    Retrieves the current git commit hash of the repository containing this file.

    This is useful for reproducibility, allowing you to track the exact version
    of the code used for a particular experiment.

    Returns:
        str | None: The git commit hash as a string if successful, otherwise None.
    """

    try:
        # Get the directory where this script is located
        source_sub_dir = os.path.dirname(os.path.realpath(__file__))
        # Execute the git command to get the current HEAD commit hash
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_sub_dir, stderr=subprocess.DEVNULL)
            .decode(sys.stdout.encoding)
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Handle cases where git is not installed or the directory is not a git repo
        logging.warning(
            "Could not retrieve git hash. This may be because the code is not in a git repository "
            "or git is not installed. Git hash checking will be ignored."
        )
        return None
    return git_hash


def write_git_hash(workdir_path: str) -> None:
    """
    Writes the current git hash to a file in a specified directory.

    If a hash file already exists, it compares the current hash with the saved one
    and logs a warning if they differ.

    Args:
        workdir_path (str): The path to the directory where the git hash file will be saved.
    """
    git_hash = check_git_hash()
    if git_hash is None:
        return

    saved_git_hash_path = os.path.join(workdir_path, GIT_HASH_NAME)
    if os.path.exists(saved_git_hash_path):
        # If hash file exists, compare it with the current hash
        with open(saved_git_hash_path) as f:
            saved_git_hash = f.read().strip()
        if saved_git_hash != git_hash:
            logging.warning(f"Git hash has changed. Saved: {saved_git_hash[:8]}, Current: {git_hash[:8]}")
    else:
        # If no hash file exists, write the current hash
        with open(saved_git_hash_path, "w") as f:
            f.write(git_hash)


def latest_checkpoint_path(dir_path: str, regex: str | None = None) -> str:
    """
    Finds the path of the latest checkpoint file or directory in a directory.

    The latest checkpoint is determined by sorting the filenames alphanumerically
    and picking the last one. This assumes a naming convention like `checkpoint_1000.pt`,
    `checkpoint_2000.pt`, etc.

    Args:
        dir_path (str): The directory to search for checkpoints.
        regex (str | None, optional): A glob pattern to match checkpoint files. If None,
                                      a default pattern is used. Defaults to None.

    Returns:
        str: The full path to the latest checkpoint file.

    Raises:
        AssertionError: If no files matching the regex are found in the directory.
    """
    if regex is None:
        regex = CHECKPOINT_FORMAT.format("*")

    f_list = glob.glob(os.path.join(dir_path, regex))
    if not f_list:
        raise FileNotFoundError(f"No checkpoint files or directories found in {dir_path} matching '{regex}'")

    # Sort files based on the integer values in their names
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

    latest_path = f_list[-1]
    logging.info(f"Latest checkpoint '{os.path.relpath(latest_path, start=dir_path)}' found in '{dir_path}'.")
    return latest_path


def manage_checkpoints(dir_path: str, max_checkpoints: int, regex: str | None = None):
    """Keeps the most recent checkpoints and deletes older ones."""
    if regex is None:
        regex = CHECKPOINT_FORMAT.format("*")

    checkpoints = glob.glob(os.path.join(dir_path, regex))

    if len(checkpoints) > max_checkpoints:
        # Sort files based on the integer values in their names
        checkpoints.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        num_to_delete = len(checkpoints) - max_checkpoints
        for old_checkpoint in checkpoints[:num_to_delete]:
            logging.info(f"Deleting old checkpoint: {old_checkpoint}")
            if os.path.isfile(old_checkpoint):
                os.remove(old_checkpoint)
            else:
                shutil.rmtree(old_checkpoint)
