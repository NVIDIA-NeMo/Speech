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

from dataclasses import dataclass
import torch

from nemo.collections.asr.parts.utils.asr_confidence_utils import (
    ConfidenceConfig,
    ConfidenceMethodConfig,
    get_confidence_aggregation_bank,
    get_confidence_measure_bank,
)
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis


# frozen is required to allow hashing of this class and use it
# as a dictionary key when running confidence tuning
@dataclass(frozen=True)
class ConfidenceSpec:
    exclude_blank: bool
    aggregation: str
    confidence_type: str
    alpha: float

    def to_confidence_config(self) -> ConfidenceConfig:
        """Converts confidence spec to the confidence config.

        Internally, the tuning procedure uses this "spec" objects as they
        are more aligned with how things are implemented. But when it's time
        to save the models or call transcribe, we need to use the proper
        object of type ``ConfidenceConfig``.
        """
        if self.confidence_type == 'max_prob':
            name = 'max_prob'
            entropy_type = 'tsallis'  # can be any
            entropy_norm = 'lin'  # can be any
        else:
            name, entropy_type, entropy_norm = self.confidence_type.split("_")
        return ConfidenceConfig(
            exclude_blank=self.exclude_blank,
            aggregation=self.aggregation,
            method_cfg=ConfidenceMethodConfig(
                name=name,
                entropy_type=entropy_type,
                alpha=self.alpha,
                entropy_norm=entropy_norm,
            ),
        )


def get_filtered_logprobs(hypothesis: Hypothesis, exclude_blank: bool) -> torch.Tensor:
    """Returns logprobs from the hypothesis object with optional blanks filter.

    This function supports both CTC and Transducer hypotheses. Will place the
    logprobs on GPU if it's available.

    Args:
        hypothesis: generated hypothesis as returned from the transcribe
            method of the ASR model.
        exclude_blank: whether to filter out all ``<blank>`` tokens.

    Returns:
        torch.Tensor: of shape [S, V], where S is (filtered) sequence length and
        V is the vocabulary size.
    """
    if isinstance(hypothesis.alignments, list):  # Transducer
        filtered_logprobs = []
        for alignment in hypothesis.alignments:
            for align_elem in alignment:
                if not exclude_blank:
                    filtered_logprobs.append(align_elem[0])
                elif align_elem[1].item() != align_elem[0].shape[-1] - 1:
                    filtered_logprobs.append(align_elem[0])
        if not filtered_logprobs:  # for the edge-case of all blanks
            filtered_logprobs.append(align_elem[0])
        filtered_logprobs = torch.stack(filtered_logprobs)
        if torch.cuda.is_available():  # by default logprobs are placed on cpu in nemo
            filtered_logprobs = filtered_logprobs.cuda()
    else:  # CTC
        logprobs = hypothesis.y_sequence
        if torch.cuda.is_available():  # by default logprobs are placed on cpu in nemo
            logprobs = logprobs.cuda()
        if exclude_blank:  # filtering blanks
            labels = logprobs.argmax(dim=-1)
            filtered_logprobs = logprobs[labels != logprobs.shape[1] - 1]
            if filtered_logprobs.shape[0] == 0:  # for the edge-case of all blanks
                filtered_logprobs = logprobs[:1]
        else:
            filtered_logprobs = logprobs

    # need to make sure logprobs are always normalized, so checking if they sum up to 1
    if not torch.allclose(filtered_logprobs[0].exp().sum(), torch.tensor(1.0)):
        filtered_logprobs = torch.log_softmax(filtered_logprobs, dim=1)

    return filtered_logprobs


def compute_confidence(hypothesis: Hypothesis, confidence_cfg: ConfidenceConfig) -> float:
    """Computes confidence score of the full utterance from a given hypothesis.

    This is essentially a re-implementation of the built-in confidence
    computation in NeMo. The difference is that we aggregate full-utterance
    scores, while core functionality only supports word and token level
    aggregations.

    Args:
        hypothesis: generated hypothesis as returned from the transcribe
            method of the ASR model.
        confidence_cfg: confidence config specifying what kind of
            method/aggregation should be used.

    Returns:
        float: confidence score.

    """
    filtered_logprobs = get_filtered_logprobs(hypothesis, confidence_cfg.exclude_blank)
    vocab_size = filtered_logprobs.shape[1]
    aggr_func = get_confidence_aggregation_bank()[confidence_cfg.aggregation]
    if confidence_cfg.method_cfg.name == "max_prob":
        conf_type = "max_prob"
        alpha = 1.0
    else:
        conf_type = f"entropy_{confidence_cfg.method_cfg.entropy_type}_{confidence_cfg.method_cfg.entropy_norm}"
        alpha = confidence_cfg.method_cfg.alpha
    conf_func = get_confidence_measure_bank()[conf_type]

    conf_value = aggr_func(conf_func(filtered_logprobs, v=vocab_size, t=alpha)).cpu().item()

    return conf_value
