#!/usr/bin/env python3
"""
Wrapper to run simulstream inference with NeMo configs.

This script handles NeMo config files (which don't have the 'type' field that
simulstream requires) by automatically adding the required fields on-the-fly.

Usage with text file:
    python run_nemo_simulstream.py \\
        --config nemo_streaming_asr_nmt.yaml \\
        --wav-list audio_list.txt \\
        --src-lang ru \\
        --tgt-lang en \\
        --metrics-log metrics.jsonl

Usage with manifest:
    python run_nemo_simulstream.py \\
        --config nemo_streaming_asr_nmt.yaml \\
        --manifest data/manifest.json \\
        --src-lang ru \\
        --tgt-lang en \\
        --metrics-log metrics.jsonl
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from omegaconf import OmegaConf
from simulstream_manifest_utils import load_manifest_audio_paths

LANGUAGE_CODES = {
    "en": "English",
    "ru": "Russian",
    "da": "Danish",
    "it": "Italian",
    "de": "German",
}


def add_simulstream_fields(cfg_path: str, override_chunk_size: float = None, src_lang: str = None, tgt_lang: str = None) -> str:
    """
    Load NeMo config and add simulstream-required fields.
    
    Reads chunk_size from NeMo config (streaming.chunk_size) unless overridden.
    Also adds detokenizer_type and latency_unit for evaluation.
    
    Args:
        cfg_path: Path to NeMo config file
        override_chunk_size: Optional override for chunk size (if None, reads from config)
        
    Returns:
        Path to temporary config file with added fields
    """
    # Load the config
    cfg = OmegaConf.load(cfg_path)
    
    if src_lang is not None:
        cfg.nmt.source_language = LANGUAGE_CODES[src_lang]
    if tgt_lang is not None:
        cfg.nmt.target_language = LANGUAGE_CODES[tgt_lang]
    # Check if 'type' field exists
    if 'type' not in cfg:
        print(f"Adding simulstream fields to config: {cfg_path}")
        
        # Get chunk size from config or use override
        if override_chunk_size is not None:
            speech_chunk_size = override_chunk_size
            print(f"  Using override chunk size: {speech_chunk_size}s")
        elif 'streaming' in cfg and 'chunk_size' in cfg.streaming:
            speech_chunk_size = cfg.streaming.chunk_size
            print(f"  Using chunk size from config: {speech_chunk_size}s")
        elif 'streaming' in cfg and 'chunk_size_in_secs' in cfg.streaming:
            speech_chunk_size = cfg.streaming.chunk_size_in_secs
            print(f"  Using chunk size from config: {speech_chunk_size}s")
        else:
            speech_chunk_size = 0.32
            print(f"  WARNING: No chunk_size found in config, using default: {speech_chunk_size}s")
        
        # Add required fields (including detokenizer for evaluation)
        simulstream_fields = OmegaConf.create({
            'type': 'nemo.collections.asr.inference.simulstream_pipeline_adapter.NeMoStreamingPipelineAdapter',
            'speech_chunk_size': speech_chunk_size,
            'detokenizer_type': 'simuleval',  # For metrics evaluation
            'latency_unit': 'word',            # For metrics evaluation
        })
        
        # Merge (simulstream fields first, then original config)
        cfg = OmegaConf.merge(simulstream_fields, cfg)
        
        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.yaml', prefix='nemo_simulstream_')
        with open(temp_path, 'w') as f:
            OmegaConf.save(cfg, f)
        
        print(f"  Created temporary config: {temp_path}")
        return temp_path
    else:
        print(f"Config already has 'type' field, using as-is: {cfg_path}")
        return cfg_path


def main():
    parser = argparse.ArgumentParser(
        description='Run simulstream inference with NeMo streaming ASR+NMT pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--config',
        required=True,
        help='Path to NeMo config file (YAML)'
    )
    
    # Audio input (either wav-list or manifest)
    audio_group = parser.add_mutually_exclusive_group(required=True)
    audio_group.add_argument(
        '--wav-list',
        help='Path to text file containing audio file paths (one per line)'
    )
    audio_group.add_argument(
        '--manifest',
        help='Path to NeMo manifest file (JSONL format with audio_filepath field)'
    )
    
    parser.add_argument(
        '--src-lang',
        required=True,
        help='Source language code (e.g., "ru", "en")'
    )
    parser.add_argument(
        '--tgt-lang',
        required=True,
        help='Target language code (e.g., "en", "es")'
    )
    
    # Optional arguments
    parser.add_argument(
        '--metrics-log',
        default='metrics.jsonl',
        help='Path to output metrics log file (default: metrics.jsonl)'
    )
    parser.add_argument(
        '--speech-chunk-size',
        type=float,
        default=None,
        help='Audio chunk size in seconds (default: read from config streaming.chunk_size)'
    )
    
    args = parser.parse_args()
    
    # Handle manifest input - extract audio paths to temporary file
    wav_list_path = args.wav_list
    temp_wav_list = None
    
    if args.manifest:
        print(f"Loading audio paths from manifest: {args.manifest}")
        audio_paths = load_manifest_audio_paths(args.manifest)
        
        if not audio_paths:
            print("Error: No audio files found in manifest", file=sys.stderr)
            return 1
        
        # Create temporary wav list file
        temp_fd, temp_wav_list = tempfile.mkstemp(suffix='.txt', prefix='wav_list_')
        with open(temp_wav_list, 'w') as f:
            for path in audio_paths:
                f.write(f"{path}\n")
        
        wav_list_path = temp_wav_list
        print(f"Created temporary wav list: {temp_wav_list}")
    
    # Add simulstream fields to config if needed
    config_path = add_simulstream_fields(args.config, args.speech_chunk_size, args.src_lang, args.tgt_lang)
    
    # Find simulstream_inference (check virtualenv first)
    simulstream_cmd = shutil.which('simulstream_inference')
    if not simulstream_cmd:
        print("\nError: simulstream_inference not found in PATH", file=sys.stderr)
        print("Checking common locations...", file=sys.stderr)
        # Try to find it in common venv locations
        possible_paths = [
            '/home/lgrigoryan/prog/nemo-fork/nemo-fork-venv/bin/simulstream_inference',
            str(Path.home() / '.local' / 'bin' / 'simulstream_inference'),
        ]
        for path in possible_paths:
            if Path(path).exists():
                simulstream_cmd = path
                print(f"Found at: {simulstream_cmd}")
                break
    
    if not simulstream_cmd:
        # Clean up temp files before exiting
        if temp_wav_list:
            try:
                Path(temp_wav_list).unlink()
            except Exception:
                pass
        return 1
    
    # Build simulstream command
    cmd = [
        simulstream_cmd,
        '--speech-processor-config', config_path,
        '--wav-list-file', wav_list_path,
        '--src-lang', args.src_lang,
        '--tgt-lang', args.tgt_lang,
        '--metrics-log-file', args.metrics_log,
    ]

    
    print(f"\n{'='*70}")
    print("Running simulstream inference with NeMo pipeline")
    print(f"{'='*70}")
    print(f"Config: {args.config}")
    if args.manifest:
        print(f"Manifest: {args.manifest}")
    else:
        print(f"Audio list: {args.wav_list}")
    print(f"Source language: {args.src_lang}")
    print(f"Target language: {args.tgt_lang}")
    print(f"Metrics output: {args.metrics_log}")
    print(f"{'='*70}\n")
    
    # Run simulstream
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\nError running simulstream: {e}", file=sys.stderr)
        return e.returncode
    except FileNotFoundError:
        print("\nError: simulstream_inference command not found.", file=sys.stderr)
        print("Make sure simulstream is installed and in your PATH.", file=sys.stderr)
        return 1
    finally:
        # Clean up temp config file if we created one
        if config_path != args.config:
            try:
                Path(config_path).unlink()
                print(f"\nCleaned up temporary config: {config_path}")
            except Exception:
                pass
        
        # Clean up temp wav list file if we created one
        if temp_wav_list:
            try:
                Path(temp_wav_list).unlink()
                print(f"Cleaned up temporary wav list: {temp_wav_list}")
            except Exception:
                pass


if __name__ == '__main__':
    sys.exit(main())

