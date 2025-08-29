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
from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import OmegaConf

from nemo.core.config import hydra_runner
from nemo.utils.model_utils import import_class_by_path


@dataclass
class HfExportConfig:
    # Name of the model class to be imported, e.g. nemo.collections.speechlm2.models.DuplexS2SModel
    class_path: str

    # Path to PyTorch Lightning checkpoint file (normal ckpt) or directory (distributed ckpt)
    ckpt_path: str

    # Path to the experiment's config, used to instantiate the model class.
    ckpt_config: str

    # Path where we should save the HuggingFace Hub compatible checkpoint
    output_dir: str

    # Dtype used for stored parameters
    dtype: str = "bfloat16"


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str):
    if Path(checkpoint_path).is_dir():
        from torch.distributed.checkpoint import load

        state_dict = {"state_dict": model.state_dict()}
        load(state_dict, checkpoint_id=checkpoint_path)
        model.load_state_dict(state_dict["state_dict"])
    else:
        ckpt_data = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt_data["state_dict"])


@hydra_runner(config_name="HfExportConfig", schema=HfExportConfig)
def main(cfg: HfExportConfig):
    """
    Read PyTorch Lightning checkpoint and export the model to HuggingFace Hub format.
    The resulting model can be then initialized via ModelClass.from_pretrained(path).

    Also supports distributed checkpoints for models trained with FSDP2/TP.
    """
    model_cfg = OmegaConf.to_container(OmegaConf.load(cfg.ckpt_config).model, resolve=True)
    model_cfg["torch_dtype"] = cfg.dtype
    cls = import_class_by_path(cfg.class_path)
    model = cls(model_cfg)
    load_checkpoint(model, cfg.ckpt_path)
    model = model.to(getattr(torch, cfg.dtype))
    model.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()
