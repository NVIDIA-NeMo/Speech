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

from abc import ABC
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from omegaconf import DictConfig

from nemo.utils import logging


@dataclass
class GraphIntersectDenseConfig:
    """Graph dense intersection config.
    """

    search_beam: float = 20.0
    output_beam: float = 10.0
    min_active_states: int = 30
    max_active_states: int = 10000


@dataclass
class GraphModuleConfig:
    """Config for graph modules.
    Typically used with graph losses and decoders.
    """

    topo_type: str = "default"
    topo_with_self_loops: bool = True
    token_lm: Optional[Any] = None
    intersect_pruned: bool = False
    intersect_conf: GraphIntersectDenseConfig = GraphIntersectDenseConfig()
    boost_coeff: float = 0.0
    predictor_window_size: int = 0
    predictor_step_size: int = 1


class ASRK2Mixin(ABC):
    """k2 Mixin class that simplifies the construction of various models with k2-based losses.
    
    It does the following:
        -   Sets up the graph loss and decoder (methods _init_k2 and update_k2_modules).
        -   Registers external graphs, if needed.
        -   Augments forward(...) with optional graph decoding to get accurate predictions.
    """

    def _init_k2(self):
        """
        k2-related initialization implementation.

        This method is expected to run after the __init__ which sets self._cfg
        self._cfg is expected to have the attribute graph_module_cfg
        """
        if not hasattr(self, "_cfg"):
            raise ValueError("self._cfg must be set before calling _init_k2().")
        if not hasattr(self._cfg, "graph_module_cfg") or self._cfg.graph_module_cfg is None:
            raise ValueError("self._cfg.graph_module_cfg must be set and cannot be None.")
        self.graph_module_cfg = self._cfg.graph_module_cfg

        # register token_lm for MAPLoss
        criterion_type = self.graph_module_cfg.get("criterion_type", "ml")
        self.use_graph_lm = criterion_type == "map"
        if self.use_graph_lm:
            token_lm_path = self.graph_module_cfg.backend_cfg.get("token_lm", None)
            if token_lm_path is None:
                raise ValueError(
                    f"graph_module_cfg.backend_cfg.token_lm is empty. It must be set for criterion_type == `{criterion_type}`"
                )
            token_lm_path = self.register_artifact('graph_module_cfg.backend_cfg.token_lm', token_lm_path)
            self.graph_module_cfg.backend_cfg["token_lm"] = token_lm_path

        self.update_k2_modules(self.graph_module_cfg)

    def update_k2_modules(self, input_cfg: DictConfig):
        """
        Helper function to initialize or update k2 loss and transcribe_decoder.

        Args:
            input_cfg: DictConfig to take new parameters from. Schema is expected as in
                nemo.collections.asr.models.configs.k2_sequence_models_config.GraphModuleConfig
        """
        del self.loss
        if hasattr(self, "transcribe_decoder"):
            del self.transcribe_decoder

        if hasattr(self, "joint"):
            # RNNT
            num_classes = self.joint.num_classes_with_blank - 1
        else:
            # CTC, MMI, ...
            num_classes = self.decoder.num_classes_with_blank - 1
            remove_consecutive = input_cfg.backend_cfg.get("topo_with_self_loops", True) and input_cfg.backend_cfg.get(
                "topo_type", "default"
            ) not in ["forced_blank", "identity",]
            self._wer.remove_consecutive = remove_consecutive

        from nemo.collections.asr.losses.lattice_losses import LatticeLoss

        self.loss = LatticeLoss(
            num_classes=num_classes,
            reduction=self._cfg.get("ctc_reduction", "mean_batch"),
            backend="k2",
            criterion_type=input_cfg.get("criterion_type", "ml"),
            loss_type=input_cfg.get("loss_type", "ctc"),
            split_batch_size=input_cfg.get("split_batch_size", 0),
            graph_module_cfg=input_cfg.backend_cfg,
        )

        criterion_type = self.loss.criterion_type
        self.use_graph_lm = criterion_type == "map"
        transcribe_training = input_cfg.get("transcribe_training", False)
        if transcribe_training and criterion_type == "ml":
            logging.warning(
                f"""You do not need to use transcribe_training=`{transcribe_training}` 
                            with criterion_type=`{criterion_type}`. transcribe_training will be set to False."""
            )
            transcribe_training = False
        self.transcribe_training = transcribe_training
        if self.use_graph_lm:
            from nemo.collections.asr.modules.graph_decoder import ViterbiDecoderWithGraph

            self.transcribe_decoder = ViterbiDecoderWithGraph(
                num_classes=num_classes,
                backend="k2",
                dec_type="token_lm",
                return_type="1best",
                return_ilabels=True,
                output_aligned=True,
                split_batch_size=input_cfg.get("split_batch_size", 0),
                graph_module_cfg=input_cfg.backend_cfg,
            )

    def _forward_k2_post_processing(
        self, log_probs: torch.Tensor, encoded_length: torch.Tensor, greedy_predictions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        k2-related post-processing parf of .forward()

        Args:
            log_probs: The log probabilities tensor of shape [B, T, D].
            encoded_length: The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
            greedy_predictions: The greedy token predictions of the model of shape [B, T]

        Returns:
            A tuple of 3 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
            3) The greedy token predictions of the model of shape [B, T] (via argmax)
        """
        # greedy_predictions from .forward() are incorrect for criterion_type=`map`
        # getting correct greedy_predictions, if needed
        if self.use_graph_lm and (not self.training or self.transcribe_training):
            greedy_predictions, encoded_length, _ = self.transcribe_decoder.forward(
                log_probs=log_probs, log_probs_length=encoded_length
            )
        return log_probs, encoded_length, greedy_predictions
