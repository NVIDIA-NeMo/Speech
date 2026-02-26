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

"""
Convert .nemo checkpoints that were trained with ``preprocessor.use_torchaudio=True``
to the current format (non-torchaudio FilterbankFeatures).

After torchaudio was removed as a dependency (PR #15211), models trained with the
torchaudio-based preprocessor (FilterbankFeaturesTA) fail to load because the
state dict keys no longer match:

    Old (torchaudio):
        preprocessor.featurizer._mel_spec_extractor.spectrogram.window
        preprocessor.featurizer._mel_spec_extractor.mel_scale.fb

    New (current):
        preprocessor.featurizer.window
        preprocessor.featurizer.fb

This script renames those keys and also sets ``use_torchaudio: false`` in the model
config so that the correct featurizer class is instantiated on load.

Usage
-----
    python convert_torchaudio_nemo.py --nemo_file model.nemo --output_file model_converted.nemo
"""

import argparse
import os
import tarfile
import tempfile

import torch
import yaml


MODEL_CONFIG_YAML = "model_config.yaml"
MODEL_WEIGHTS_CKPT = "model_weights.ckpt"

# Old torchaudio key suffix -> new key suffix
KEY_MIGRATION = {
    "featurizer._mel_spec_extractor.spectrogram.window": "featurizer.window",
    "featurizer._mel_spec_extractor.mel_scale.fb": "featurizer.fb",
}


def migrate_state_dict(state_dict: dict) -> tuple[dict, list[tuple[str, str]]]:
    """Rename torchaudio-era keys.  Returns (new_state_dict, list of (old, new) renames)."""
    renames = []
    for key in list(state_dict.keys()):
        for old_suffix, new_suffix in KEY_MIGRATION.items():
            if key.endswith(old_suffix):
                new_key = key[: -len(old_suffix)] + new_suffix
                if "featurizer.fb" in new_suffix:
                    state_dict[new_key] = state_dict.pop(key).T.unsqueeze(0)
                else:
                    state_dict[new_key] = state_dict.pop(key)
                renames.append((key, new_key))
                break
    return state_dict, renames


def migrate_config(cfg: dict) -> bool:
    """Set ``use_torchaudio: false`` in the preprocessor config.  Returns True if changed."""
    preprocessor = cfg.get("preprocessor", {})
    if preprocessor.get("use_torchaudio", False):
        preprocessor["use_torchaudio"] = False
        return True
    return False


def convert_nemo_file(nemo_path: str, output_path: str) -> None:
    """Extract, migrate, and repack a .nemo archive."""
    with tempfile.TemporaryDirectory() as tmpdir:

        def _safe_extract_all(tar_obj: tarfile.TarFile, dest_dir: str) -> None:
            """Safely extract all members of a tar file into dest_dir.

            Ensures that no member escapes dest_dir via absolute paths or '..' components.
            """
            dest_dir_abs = os.path.abspath(dest_dir)
            for member in tar_obj.getmembers():
                member_path = os.path.join(dest_dir_abs, member.name)
                member_path_abs = os.path.abspath(member_path)
                if os.path.commonpath([dest_dir_abs, member_path_abs]) != dest_dir_abs:
                    raise ValueError(f"Illegal tar archive entry path: {member.name!r}")
                tar_obj.extract(member, path=dest_dir_abs)

        # --- Unpack --------------------------------------------------------
        # Older checkpoints may be gzipped; newer ones are plain tar.
        try:
            tar = tarfile.open(nemo_path, "r:")
        except tarfile.ReadError:
            tar = tarfile.open(nemo_path, "r:gz")
        _safe_extract_all(tar, tmpdir)
        tar.close()

        # --- Migrate state dict --------------------------------------------
        weights_path = os.path.join(tmpdir, MODEL_WEIGHTS_CKPT)
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"Could not find {MODEL_WEIGHTS_CKPT} inside the .nemo archive. "
                "Are you sure this is a valid .nemo file?"
            )

        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        state_dict, renames = migrate_state_dict(state_dict)
        if not renames:
            print("No torchaudio keys found in state dict — nothing to migrate.")
            return

        for old, new in renames:
            print(f"  Renamed: {old}  ->  {new}")

        torch.save(state_dict, weights_path)

        # --- Migrate config ------------------------------------------------
        config_path = os.path.join(tmpdir, MODEL_CONFIG_YAML)
        if os.path.isfile(config_path):
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            if migrate_config(cfg):
                print("  Config:  set use_torchaudio=false")
                with open(config_path, "w") as f:
                    yaml.dump(cfg, f, default_flow_style=False)

        # --- Repack --------------------------------------------------------
        with tarfile.open(output_path, "w:") as tar:
            tar.add(tmpdir, arcname=".")

    print(f"\nConverted checkpoint saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert .nemo checkpoints from torchaudio preprocessor format to the current format.",
    )
    parser.add_argument(
        "--nemo_file",
        required=True,
        help="Path to the source .nemo file.",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Path to write the converted .nemo file.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.nemo_file):
        raise FileNotFoundError(f"File not found: {args.nemo_file}")

    convert_nemo_file(args.nemo_file, args.output_file)


if __name__ == "__main__":
    main()
