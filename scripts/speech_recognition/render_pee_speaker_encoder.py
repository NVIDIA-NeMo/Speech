#!/usr/bin/env python3
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

"""Render the dense speaker Transformer from a PEE .nemo bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from pathlib import Path

from omegaconf import OmegaConf
from safetensors.torch import save_file

from nemo.collections.asr.modules.parallel_expert_encoder_ggemm import ParallelExpertEncoderPT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def read_speaker_config(bundle: Path):
    with tarfile.open(bundle, mode="r") as archive:
        members = [member for member in archive.getmembers() if os.path.basename(member.name) == "model_config.yaml"]
        if len(members) != 1:
            raise RuntimeError(f"Expected one model_config.yaml in {bundle}, found {len(members)}.")
        stream = archive.extractfile(members[0])
        if stream is None:
            raise RuntimeError(f"Could not read model_config.yaml from {bundle}.")
        bundle_config = OmegaConf.create(stream.read().decode("utf-8"))
    config = bundle_config.get("speaker_expert_cfg", None)
    if config is None or not str(config.get("_target_", "")).endswith("TransformerEncoder"):
        raise ValueError(f"{bundle} has no regular TransformerEncoder speaker_expert_cfg.")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Source ParallelExpertEncoderPT .nemo bundle",
    )
    parser.add_argument("--output", type=Path, required=True, help="New standalone artifact directory")
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = read_speaker_config(source)

    pee = ParallelExpertEncoderPT.load_from_nemo(str(source), map_location="cpu", strict=True)
    speaker = pee.pee.experts["speaker"]
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in speaker.state_dict().items()}
    save_file(state, str(output / "model.safetensors"))
    OmegaConf.save(config, output / "model_config.yaml")

    metadata = {
        "format": "nemo-standalone-transformer-encoder-v1",
        "source": str(source),
        "source_sha256": sha256(source),
        "source_expert": "speaker",
        "parameters": sum(parameter.numel() for parameter in speaker.parameters()),
        "dtypes": sorted({str(tensor.dtype) for tensor in state.values()}),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
