#!/usr/bin/env python3
"""
Simple wrapper to run simulstream_score_quality from a NeMo manifest.

Usage:
    python run_simulstream_comet_from_manifest.py \
        --manifest /path/to/manifest.json \
        --metrics-log metrics.jsonl \
        --config config.yaml
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import yaml

from simulstream_manifest_utils import manifest_to_audio_definition


def extract_from_manifest(manifest_path):
    """Extract audio definitions, references, and transcripts from manifest."""
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
    
    return audio_defs, references, transcripts


def main():
    parser = argparse.ArgumentParser(
        description='Run simulstream COMET evaluation from NeMo manifest'
    )
    
    parser.add_argument(
        '--manifest',
        required=True,
        help='Path to NeMo manifest JSONL file'
    )
    parser.add_argument(
        '--metrics-log',
        required=True,
        help='Path to metrics JSONL log file (from inference)'
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to NeMo/simulstream config YAML file'
    )
    parser.add_argument(
        '--scorer',
        default='comet',
        help='Scorer to use (default: comet)'
    )
    parser.add_argument(
        '--model',
        default='Unbabel/wmt22-comet-da',
        help='COMET model (default: Unbabel/wmt22-comet-da)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='COMET batch size (default: 16)'
    )
    
    args = parser.parse_args()
    
    # Extract data from manifest
    print(f"Reading manifest: {args.manifest}")
    audio_defs, references, transcripts = extract_from_manifest(args.manifest)
    print(f"  Found {len(audio_defs)} audio files")
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_def_file, refs_file, trans_file = manifest_to_audio_definition(args.manifest, tmpdir)
        
        # Run simulstream_score_quality
        cmd = [
            'simulstream_score_quality',
            '--eval-config', args.config,
            '--log-file', args.metrics_log,
            '--audio-definition', str(audio_def_file),
            '--references', str(refs_file),
            '--transcripts', str(trans_file),
            '--scorer', args.scorer,
            '--model', args.model,
            '--batch-size', str(args.batch_size),
        ]
        
        print(f"\nRunning simulstream_score_quality...")
        print(f"Command: {' '.join(cmd)}\n")
        
        try:
            result = subprocess.run(cmd, check=True)
            return result.returncode
        except subprocess.CalledProcessError as e:
            print(f"\nError: simulstream_score_quality failed with exit code {e.returncode}", 
                  file=sys.stderr)
            return e.returncode
        except FileNotFoundError:
            print("\nError: simulstream_score_quality not found. Is simulstream installed?",
                  file=sys.stderr)
            return 1


if __name__ == '__main__':
    sys.exit(main())

