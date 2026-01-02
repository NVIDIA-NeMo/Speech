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

import difflib
import os
from typing import List

import nemo_run as run
from lightning.pytorch.callbacks.callback import Callback
from nemo_run.core.serialization.yaml import YamlSerializer
from nemo_run.run.torchx_backend.packaging import _serialize

from nemo.collections.common.tokenizers.huggingface import AutoTokenizer
from nemo.collections.llm.gpt.data.squad import SquadDataModule
from nemo.collections.llm.gpt.model import GPTModel
from nemo.collections.llm.recipes.llama3_8b import MegatronCommOverlapCallback
from nemo.lightning.base import DEFAULT_NEMO_CACHE_HOME
from nemo.utils import logging

DEFAULT_NEMO_HOME = os.getenv('NEMO_HOME', DEFAULT_NEMO_CACHE_HOME)


def hf_tokenizer(model_name: str) -> run.Config[AutoTokenizer]:
    """
    HuggingFace tokenizer.

    Args:
        model_name (str): corresponds to HuggingFace-AutoTokenizer's 'pretrained_model_name_or_path' input argument.
                For more details please refer to-
                huggingface.co/docs/transformers/v4.47.1/en/model_doc/auto#transformers.AutoTokenizer
    """
    log_msg = [
        f"`AutoTokenizer` first searches for tokenizer files locally stored in {DEFAULT_NEMO_HOME}.",
        "(from env var `NEMO_HOME`- can be changed using '-nh/--nemo_home' CLI arg).",
        "If files are missing locally, `AutoTokenizer` will try downloading from HuggingFace. In this case-",
        "make sure env vars 'HF_HUB_OFFLINE':'0' and 'HF_TOKEN':'<token_value>' are set in your sbatch script.",
        "Both of these will be set automatically if you provide '-hf/--hf_token' CLI arg.",
    ]
    logging.warning(" ".join(log_msg))

    return run.Config(
        AutoTokenizer,
        pretrained_model_name=model_name,
        use_fast=True,
    )


def import_ckpt_experiment(executor: run.SlurmExecutor, model: run.Config[GPTModel], source: str):
    """
    Downloads/Acceses checkpoint to be used for fine-tuning. `import_ckpt` first tries find the nemo checkpoint in
    <NEMO_HOME>/models/. For eg: for llama3 8b, the path will look like- <NEMO_HOME>/models/meta-llama/Meta-Llama-3-8B
    If missing, tries to downloads at the same location from HuggingFace and converts it nemo format.

    Args:
        source (str): HuggingFace URL. For eg- hf://meta-llama/Meta-Llama-3-70B
    """
    from copy import deepcopy

    from nemo.collections.llm import import_ckpt

    import_executor = deepcopy(executor)
    import_executor.ntasks_per_node = 1
    import_executor.nodes = 1

    return run.Partial(import_ckpt, model=model, source=source, overwrite=False), import_executor, "import_ckpt_exp"


def get_nemo_home(nemo_home=None):
    """
    Get NEMO_HOME path. Checks for both nemo_home argument and NEMO_HOME environment variable.
    """
    arg_nemo_set = nemo_home is True
    env_nemo_set = "NEMO_HOME" in os.environ

    if arg_nemo_set and env_nemo_set:
        if os.environ["NEMO_HOME"] != nemo_home:
            logging.warning(f"Using nemo_home ({nemo_home}) instead of NEMO_HOME ({os.environ['NEMO_HOME']})")
        return nemo_home

    if arg_nemo_set:
        return nemo_home

    if env_nemo_set:
        return os.environ["NEMO_HOME"]

    raise ValueError("Neither -nh/--nemo_home argument nor NEMO_HOME environment variable is set")


def prepare_squad_dataset(model_name: str, seq_length: int = 2048, nemo_home=None):
    """Prepare the SQuAD dataset for fine-tuning.

    Args:
        model_name (str): The name of the model
        seq_length (int): The sequence length to use for packing. Defaults to 2048.
        nemo_home: Optional path to NEMO home directory set via args.nemo_home
    """
    from pathlib import Path

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.llm.gpt.data.packed_sequence import PackedSequenceSpecs
    from nemo.collections.llm.gpt.data.squad import SquadDataModule

    nemo_home_path = Path(get_nemo_home(nemo_home))
    dataset_root = nemo_home_path / "datasets" / "squad"
    dataset_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer(pretrained_model_name=model_name)

    # Configure SquadDataModule with packing specs
    datamodule = SquadDataModule(
        dataset_root=dataset_root,
        seq_length=seq_length,
        global_batch_size=8,
        micro_batch_size=1,
        packed_sequence_specs=PackedSequenceSpecs(packed_sequence_size=seq_length),
        tokenizer=tokenizer,
        force_redownload=True,
        delete_raw=False,
        seed=1234,
    )

    # This will generate both JSONL and packed .bin files
    datamodule.prepare_data()

    # Verify the output
    packed_dir = dataset_root / "packed" / model_name.replace("/", "--")
    print(f"Packed files should be in: {packed_dir}")
    if packed_dir.exists():
        print("Files found:", list(packed_dir.glob("*")))
    else:
        raise FileNotFoundError(f"Packed dataset dir not found at {packed_dir}. Dataset download failed")


def prepare_squad_dataset_experiment(
    executor: run.SlurmExecutor, model_name: str, seq_length: int = 2048, nemo_home=None
):
    """
    Downloads and prepares the SQuAD dataset for fine-tuning.
    """
    from copy import deepcopy

    dataset_executor = deepcopy(executor)
    dataset_executor.ntasks_per_node = 1
    dataset_executor.nodes = 1

    return (
        run.Partial(
            prepare_squad_dataset,
            model_name=model_name,
            seq_length=seq_length,
            nemo_home=nemo_home,
        ),
        dataset_executor,
        "prepare_squad_dataset_exp",
    )


def isfile_train_pack_metadata(hf_model_uri: str, data_config: run.Config[SquadDataModule]) -> bool:
    """
    This method is used for fine-tuning. It checks if packed train data for a partiular
    sequence length exists locally. This is needed to set data flag (force_redownload=True)
    which avoids experiment crash in case files are missing.
    """
    datasets_dir = os.getenv("NEMO_DATASETS_CACHE", os.path.join(DEFAULT_NEMO_HOME, "datasets"))
    model_dir = hf_model_uri.replace("/", "--")
    metadata_filename = f"{data_config.seq_length}_metadata.jsonl"

    train_pack_metadata_filepath = os.path.join(datasets_dir, "squad", "packed", model_dir, metadata_filename)

    return os.path.exists(train_pack_metadata_filepath) and os.path.isfile(train_pack_metadata_filepath)


def get_comm_overlap_callback_idx(callbacks: List[Callback]) -> int | None:
    """
    nemo.lightning.Trainer has a list of callbacks defined. This method identifies index of MegatronCommOverlapCallback
    from the list defined in recipes in nemo.collections.llm.recipes. The index is needed to override ddp communication
    params
    """
    if callbacks:  # default is None in lightning
        for idx, callback in enumerate(callbacks):
            if callback.__fn_or_cls__ == MegatronCommOverlapCallback:
                return idx
    return None


def dump_config_diff_from_base_recipe(
    base_recipe: str, new_recipe: str, output_dir: str, file_name: str = "config_diff.txt"
):
    """
    Dump the config diff from the base recipe.
    """
    base_recipe_config = _serialize(base_recipe, serializer_cls=YamlSerializer)
    new_recipe_config = _serialize(new_recipe, serializer_cls=YamlSerializer)
    diff = difflib.unified_diff(
        base_recipe_config.splitlines(keepends=True),
        new_recipe_config.splitlines(keepends=True),
        fromfile="base_recipe",
        tofile="new_recipe",
        lineterm="",
    )
    diff = "".join(diff)
    print("dumping config diff to ", os.path.join(output_dir, file_name))
    with open(os.path.join(output_dir, file_name), "w") as f:
        f.write(diff)
