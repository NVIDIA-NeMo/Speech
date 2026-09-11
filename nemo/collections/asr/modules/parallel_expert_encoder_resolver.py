# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Schema-based dispatch for the two public Parallel Expert Encoder families.

The original two-branch encoder already shipped on ``main`` under
``parallel_expert_encoder.ParallelExpertEncoder``.  The grouped-GEMM encoder is
intentionally published from a different module.  Historical archives can share
the same target basename, so their architecture is determined from the embedded
configuration rather than from the target string alone.
"""

from __future__ import annotations

import os
import tarfile
from typing import Literal, Mapping

from omegaconf import DictConfig, OmegaConf

ParallelExpertEncoderKind = Literal["two_branch", "ggemm"]

_TWO_BRANCH_KEYS = frozenset(("asr_encoder_cfg", "diarization_model_cfg"))
_GGEMM_KEYS = frozenset(("speech_expert_cfg", "speaker_expert_cfg", "sound_expert_cfg", "sortformer_modules_cfg"))


def classify_parallel_expert_encoder_config(config: Mapping) -> ParallelExpertEncoderKind:
    """Return the unique encoder family described by ``config``.

    A partially specified or ambiguous schema is rejected instead of falling
    through to whichever class happened to be imported first.
    """

    keys = set(config.keys())
    has_two_branch = _TWO_BRANCH_KEYS.issubset(keys) and all(
        config.get(key) not in (None, {}, "") for key in _TWO_BRANCH_KEYS
    )
    has_ggemm = _GGEMM_KEYS.issubset(keys) and all(config.get(key) not in (None, {}, "") for key in _GGEMM_KEYS)
    if has_two_branch == has_ggemm:
        raise ValueError(
            "ParallelExpertEncoder config must describe exactly one architecture: "
            "two-branch requires asr_encoder_cfg + diarization_model_cfg; grouped-GEMM "
            "requires speech/speaker/sound expert configs + sortformer_modules_cfg."
        )
    return "two_branch" if has_two_branch else "ggemm"


def read_parallel_expert_encoder_bundle_config(nemo_path: str) -> DictConfig:
    """Read and validate ``model_config.yaml`` from a local ``.nemo`` archive."""

    if not (isinstance(nemo_path, str) and nemo_path.endswith(".nemo") and os.path.isfile(nemo_path)):
        raise ValueError(f"Expected an existing local .nemo archive, got {nemo_path!r}.")
    try:
        with tarfile.open(nemo_path, mode="r") as archive:
            members = [
                member
                for member in archive.getmembers()
                if member.isfile() and os.path.basename(member.name) == "model_config.yaml"
            ]
            if len(members) != 1:
                raise ValueError(f"{nemo_path!r} must contain exactly one model_config.yaml; found {len(members)}.")
            stream = archive.extractfile(members[0])
            if stream is None:
                raise ValueError(f"Could not read model_config.yaml from {nemo_path!r}.")
            return OmegaConf.create(stream.read().decode("utf-8"))
    except (tarfile.TarError, OSError, UnicodeDecodeError) as error:
        raise ValueError(f"Could not inspect ParallelExpertEncoder bundle {nemo_path!r}: {error}") from error


def resolve_parallel_expert_encoder_pt(
    model_path_or_name: str | None = None,
    *,
    config: Mapping | None = None,
    architecture: str | None = None,
):
    """Resolve the ModelPT shell for a local bundle, inline config, or remote id.

    Remote identifiers cannot be inspected without first downloading them.  To
    preserve the public behavior already released on ``main``, an unspecified
    remote identifier continues to mean the two-branch architecture.  New
    grouped-GEMM remote identifiers must set ``architecture='ggemm'``.
    """

    if config is not None:
        kind = classify_parallel_expert_encoder_config(config)
    elif model_path_or_name and model_path_or_name.endswith(".nemo") and os.path.isfile(model_path_or_name):
        kind = classify_parallel_expert_encoder_config(read_parallel_expert_encoder_bundle_config(model_path_or_name))
    else:
        normalized = None if architecture is None else str(architecture).strip().lower().replace("-", "_")
        aliases = {
            None: "two_branch",
            "two_branch": "two_branch",
            "phpee": "two_branch",
            "placeholder_pee": "two_branch",
            "ggemm": "ggemm",
            "pee": "ggemm",
            "ggemm_pee": "ggemm",
        }
        if normalized not in aliases:
            raise ValueError(
                f"Unknown ParallelExpertEncoder architecture {architecture!r}; expected two_branch/phpee or ggemm."
            )
        kind = aliases[normalized]

    if kind == "two_branch":
        from nemo.collections.asr.modules.parallel_expert_encoder import ParallelExpertEncoderPT

        return ParallelExpertEncoderPT

    from nemo.collections.asr.modules.parallel_expert_encoder_ggemm import ParallelExpertEncoderPT

    return ParallelExpertEncoderPT


def is_parallel_expert_encoder(module) -> bool:
    """Whether ``module`` explicitly advertises the shared SALM speaker contract."""

    return bool(getattr(module, "supports_external_speaker_targets", False)) and callable(
        getattr(module, "online_inference", None)
    )
