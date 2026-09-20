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
Utilities for using NeMo manifest files with simulstream evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from nemo.utils import logging


def load_manifest_audio_paths(manifest_path: str | Path) -> list[str]:
    """
    Load audio file paths from a NeMo manifest file.

    Args:
        manifest_path: Path to NeMo manifest JSONL file

    Returns:
        List of audio file paths
    """
    audio_paths = []
    manifest_dir = Path(manifest_path).parent

    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse line {line_num} in manifest: {e}")
                continue
            audio_path = data.get('audio_filepath', data.get('audio_file'))
            if audio_path:
                audio_path = Path(audio_path)
                if not audio_path.is_absolute():
                    audio_path = manifest_dir / audio_path
                audio_paths.append(str(audio_path.resolve()))

    logging.info(f"Loaded {len(audio_paths)} audio files from manifest: {manifest_path}")
    return audio_paths


def manifest_to_audio_definition(manifest_path: str | Path, output_path: str | Path) -> tuple[Path, Path, Path]:
    """
    Create simulstream audio definition YAML from a NeMo manifest, along with plaintext
    reference/transcript files. This is needed for simulstream's score/latency metrics evaluation.

    Args:
        manifest_path: Path to NeMo manifest file.
        output_path: Directory to write the generated files into.

    Returns:
        Tuple of (audio_definitions.yaml, references.txt, transcripts.txt) paths.
    """
    audio_defs = []
    references = []
    transcripts = []

    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line.strip())

            audio_path = data['audio_filepath']
            duration = data.get('duration', 0.0)
            audio_defs.append({'wav': audio_path, 'offset': 0.0, 'duration': float(duration) if duration else 0.0})

            transcripts.append(data.get('text', ''))
            # Prefer 'target_text', falling back to 'answer' (common NeMo AST manifest field).
            references.append(data.get('target_text', data.get('answer', '')))

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_def_file = output_dir / 'audio_definitions.yaml'
    with open(audio_def_file, 'w', encoding='utf-8') as f:
        yaml.dump(audio_defs, f, default_flow_style=False, allow_unicode=True)

    refs_file = output_dir / 'references.txt'
    with open(refs_file, 'w', encoding='utf-8') as f:
        f.writelines(ref + '\n' for ref in references)

    trans_file = output_dir / 'transcripts.txt'
    with open(trans_file, 'w', encoding='utf-8') as f:
        f.writelines(trans + '\n' for trans in transcripts)

    logging.info(f"Created simulstream audio definition files in: {output_dir}")
    return audio_def_file, refs_file, trans_file
