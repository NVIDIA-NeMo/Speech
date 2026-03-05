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
import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities import CombinedLoader
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.common.data.fallback import FallbackDataset
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.tokenizers import TokenizerSpec


class DataModule(LightningDataModule):
    """
    A Lightning DataModule specialized for Lhotse dataloading.
    It takes care of setting up the proper DP ranks for dataloaders, and instantiating them.
    Keep in mind the actual dataset paths and blend are defined by the YAML config, not Python code.

    The typical structure of the YAML config used to initialize this module looks like the following:

    .. code-block:: yaml

        data:
          train_ds:
            input_cfg: path/to/input_cfg.yaml
            num_workers: 2
            batch_size: 4
            # ... Other settings, see nemo/collections/common/data/lhotse/dataloader.py

          validation_ds:
            # The entries under 'datasets' are a list of separate dataloaders.
            # The structure is <dataset-name>: {<dataloader-dict-config>}
            # They inherit all settings from validation_ds, but can individually override them.
            datasets:
              val_set_0:  # rename to your dataset name, add more as needed
                cuts_path: ???  # needs to be specified
            batch_size: 4
            # ... Other settings, see nemo/collections/common/data/lhotse/dataloader.py

    See also the examples in ``examples/speechlm2/conf``.

    Args:
        cfg: a DictConfig instance, typically corresponding to `data` namespace in YAML configs.
        tokenizer: a tokenizer instance, typically NeMo's AutoTokenizer wrapping HF's AutoTokenizer.
        dataset: a torch.utils.data.Dataset instance, expected to define __getitem__ that accepts
            a lhotse.CutSet. It converts metadata + raw data to a batch of PyTorch tensors.
            The data sampling is controlled by Lhotse samplers rather than the dataset.
    """

    def __init__(self, cfg, tokenizer: TokenizerSpec, dataset: torch.utils.data.Dataset) -> None:
        super().__init__()
        self.cfg = cfg
        with open_dict(self.cfg):
            for k in ("validation_ds", "test_ds"):
                if k in self.cfg:
                    getattr(self.cfg, k).force_finite = True
                    getattr(self.cfg, k).force_map_dataset = True
        self.tokenizer = tokenizer
        self.dataset = dataset

    def train_dataloader(self):
        if "train_ds" not in self.cfg:
            return None
        return get_lhotse_dataloader_from_config(
            config=self.cfg.train_ds,
            global_rank=self._get_dp_rank(),
            world_size=self._get_world_size(),
            dataset=FallbackDataset(self.dataset),
            tokenizer=self.tokenizer,
        )

    def val_dataloader(self):
        if "validation_ds" not in self.cfg:
            return None
        cfg = self.cfg.validation_ds
        return self._build_test_dataloader(cfg)

    def test_dataloader(self):
        if "test_ds" not in self.cfg:
            return None
        cfg = self.cfg.test_ds
        return self._build_test_dataloader(cfg)

    def predict_dataloader(self):
        if "predict_ds" not in self.cfg:
            return None
        cfg = self.cfg.predict_ds

        base_cfg = cfg.copy()
        with open_dict(base_cfg):
            del base_cfg.datasets
        dloaders = {}
        for name, item in cfg.datasets.items():
            with open_dict(base_cfg):
                item = OmegaConf.merge(base_cfg, item)
            dloaders[name] = self._build_test_dataloader(item)
        # NOTE(yifan): `trainer.predict()` only supports the `CombinedLoader(mode="sequential")` mode
        # so we cannot reuse the `_build_test_dataloader` function here
        return CombinedLoader(dloaders, mode="sequential")

    def _build_test_dataloader(self, cfg: DictConfig) -> torch.utils.data.DataLoader | CombinedLoader:
        # Single validation/test dataloader.
        # This is internal-only: the config has to specify multiple dataloaders via "datasets" key,
        # even for a single validation/test set.
        if "datasets" not in cfg:
            with open_dict(cfg):
                cfg.force_finite = True
                cfg.force_map_dataset = True
            return get_lhotse_dataloader_from_config(
                config=cfg,
                global_rank=self._get_dp_rank(),
                world_size=self._get_world_size(),
                dataset=self.dataset,
                tokenizer=self.tokenizer,
            )

        # Multiple validation/test dataloaders.
        # Config looks like:
        #
        # validation_ds:
        #   batch_size: ...
        #   datasets:
        #     easy_benchmark:
        #       shar_path: ...
        #     hard_benchmark:
        #       shar_path: ...
        base_cfg = cfg.copy()
        with open_dict(base_cfg):
            del base_cfg.datasets
        dloaders = {}
        for name, item in cfg.datasets.items():
            with open_dict(base_cfg):
                item = OmegaConf.merge(base_cfg, item)
            dloaders[name] = self._build_test_dataloader(item)
        return CombinedLoader(dloaders, mode="max_size")

    def _get_dp_rank(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if (
                hasattr(self.trainer, "model")
                and hasattr(self.trainer.model, "device_mesh")
                and self.trainer.model.device_mesh is not None
            ):  # model parallelism
                return self.trainer.model.device_mesh.get_coordinate()[0]
            else:
                return torch.distributed.get_rank()  # plain ol' DDP
        else:
            return 0  # 1 GPU

    def _get_world_size(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if (
                hasattr(self.trainer, "model")
                and hasattr(self.trainer.model, "device_mesh")
                and self.trainer.model.device_mesh is not None
            ):  # model parallelism
                return self.trainer.model.device_mesh.shape[0]
            else:  # plain ol' DDP
                return torch.distributed.get_world_size()
        else:
            return 1  # 1 GPU
