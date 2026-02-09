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

"""
Utilities for using NeMo manifest files with simulstream evaluation.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def load_manifest_audio_paths(manifest_path: str) -> list[str]:
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
                audio_path = data.get('audio_filepath', data.get('audio_file'))
                if audio_path:
                    # Handle relative paths
                    audio_path = Path(audio_path)
                    if not audio_path.is_absolute():
                        audio_path = manifest_dir / audio_path
                    audio_paths.append(str(audio_path.resolve()))
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num} in manifest: {e}", file=sys.stderr)
                continue
    
    print(f"Loaded {len(audio_paths)} audio files from manifest")
    return audio_paths

def manifest_to_audio_definition(manifest_path: str, output_path: str) -> int:
    """
    Create simulstream audio definition YAML from NeMo manifest.
    
    This is needed for score/latency metrics evaluation.
    
    Args:
        manifest_path (str): Path to NeMo manifest file.
        output_path (str): Path to output YAML file.
    """
    audio_defs = []
    references = []
    transcripts = []
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            data = json.loads(line.strip())
            
            # Audio filepath
            audio_path = data['audio_filepath']
            
            # Duration (if available, otherwise use 0)
            duration = data.get('duration', 0.0)
            
            # Audio definition entry
            audio_defs.append({
                'wav': audio_path,
                'offset': 0.0,
                'duration': float(duration) if duration else 0.0
            })
            
            # Source text (English transcription)
            transcripts.append(data.get('text', ''))
            
            # Target text (Russian translation)
            # Try 'target_text' first, then 'answer', then empty
            target = data.get('target_text', data.get('answer', ''))
            references.append(target)
    
    
    audio_def_file = Path(output_path) / 'audio_definitions.yaml'
    with open(audio_def_file, 'w', encoding='utf-8') as f:
        yaml.dump(audio_defs, f, default_flow_style=False, allow_unicode=True)
    print(f"Created: {audio_def_file}")
    
    refs_file = Path(output_path) / 'references.txt'
    with open(refs_file, 'w', encoding='utf-8') as f:
        for ref in references:
            f.write(ref + '\n')
    print(f"Created: {refs_file}")
    
    trans_file = Path(output_path) / 'transcripts.txt'
    with open(trans_file, 'w', encoding='utf-8') as f:
        for trans in transcripts:
            f.write(trans + '\n')
    print(f"Created: {trans_file}")
    
    return audio_def_file, refs_file, trans_file
    




