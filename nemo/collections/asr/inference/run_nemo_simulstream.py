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
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from omegaconf import OmegaConf
from simulstream_manifest_utils import load_manifest_audio_paths

# ISO 639-1 code -> full name for NeMo NMT config. Unknown codes are passed through (no KeyError).
LANGUAGE_CODES = {
    "en": "English",
    "ru": "Russian",
    "da": "Danish",
    "it": "Italian",
    "de": "German",
    "zh": "Chinese",
}

LANGUAGE_CODE_TO_LATENCY_UNIT = {
    "en": "word",
    "ru": "word",
    "da": "word",
    "it": "word",
    "de": "word",
    "zh": "char",
}

def get_language_name(code: str) -> str:
    """Map language code to full name for config; unknown codes are returned as-is."""
    return LANGUAGE_CODES.get(code, code)


def get_latency_unit(code: str) -> str:
    """Map language code to latency unit for metrics; unknown codes default to 'word'."""
    return LANGUAGE_CODE_TO_LATENCY_UNIT.get(code, "word")


def add_simulstream_fields(cfg_path: str, output_dir: str, src_lang: str = None, tgt_lang: str = None, overrides: list = None) -> str:
    """
    Load NeMo config and add simulstream-required fields.

    Simulstream speech_chunk_size is always taken from the NeMo config (streaming.chunk_size for buffered decoding
    or streaming.chunk_size_in_secs for cache-aware models). Also adds detokenizer_type and latency_unit for evaluation.
    The generated config is saved in output_dir (e.g. same directory as the metrics log).

    Args:
        cfg_path: Path to NeMo config file
        output_dir: Directory to save the generated config (e.g. parent of metrics log)
        src_lang: Source language code
        tgt_lang: Target language code
        overrides: List of "key=value" strings to override config fields

    Returns:
        Path to the saved config file with added fields
    """
    # Load the config
    cfg = OmegaConf.load(cfg_path)
    
    # Apply command-line overrides
    if overrides:
        print("Applying command-line overrides:")
        try:
            override_conf = OmegaConf.from_dotlist(overrides)
            cfg = OmegaConf.merge(cfg, override_conf)
            for ov in overrides:
                print(f"  {ov}")
        except Exception as e:
            print(f"  Error applying overrides {overrides}: {e}")

    if src_lang is not None:
        cfg.nmt.source_language = get_language_name(src_lang)
    if tgt_lang is not None:
        cfg.nmt.target_language = get_language_name(tgt_lang)
    # Check if 'type' field exists
    if 'type' not in cfg:
        print(f"Adding simulstream fields to config: {cfg_path}")
        
        # Simulstream chunk size must match NeMo config
        if 'streaming' in cfg and 'chunk_size' in cfg.streaming:
            speech_chunk_size = cfg.streaming.chunk_size
            print(f"  Using chunk size from config: {speech_chunk_size}s for buffered decoding")
        elif 'streaming' in cfg and 'att_context_size' in cfg.streaming:
            speech_chunk_size = (cfg.streaming.att_context_size[1] + 1) * 0.08
            print(f"  Using chunk size calculated from att_context_size: {speech_chunk_size}s")
        else:
            raise ValueError(f"No chunk_size or att_context_size found in config: {cfg_path}")
        
        # Add required fields (including detokenizer for evaluation)
        simulstream_fields = OmegaConf.create({
            'type': 'nemo.collections.asr.inference.simulstream_pipeline_adapter.NeMoStreamingPipelineAdapter',
            'speech_chunk_size': speech_chunk_size,
            'detokenizer_type': 'simuleval',  # For metrics evaluation
            'latency_unit': get_latency_unit(tgt_lang),  # For metrics evaluation
        })
        
        # Merge (simulstream fields first, then original config)
        cfg = OmegaConf.merge(simulstream_fields, cfg)

        # Save in same directory as output metrics log
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = Path(cfg_path).stem + '_simulstream.yaml'
        out_path = out_dir / out_name
        with open(out_path, 'w') as f:
            OmegaConf.save(cfg, f)

        print(f"  Saved config: {out_path}")
        return str(out_path)
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
    args, unknown_args = parser.parse_known_args()

    # Unknown args of the form key=value are passed as config overrides
    overrides = []
    for arg in unknown_args:
        if arg.startswith("--"):
            print(f"Warning: Unknown argument: {arg}", file=sys.stderr)
        elif "=" in arg:
            overrides.append(arg)
        else:
            print(f"Warning: Ignoring unknown argument (expected key=value): {arg}", file=sys.stderr)

    wav_list_path = args.wav_list
    temp_wav_list = None
    try:
        if args.manifest:
            print(f"Loading audio paths from manifest: {args.manifest}")
            audio_paths = load_manifest_audio_paths(args.manifest)
            if not audio_paths:
                print("Error: No audio files found in manifest", file=sys.stderr)
                return 1
            _, temp_wav_list = tempfile.mkstemp(suffix='.txt', prefix='wav_list_')
            with open(temp_wav_list, 'w') as f:
                for path in audio_paths:
                    f.write(f"{path}\n")
            wav_list_path = temp_wav_list
            print(f"Created temporary wav list: {temp_wav_list}")

        metrics_log_dir = str(Path(args.metrics_log).parent)
        config_path = add_simulstream_fields(
            args.config, metrics_log_dir, args.src_lang, args.tgt_lang, overrides
        )

        simulstream_cmd = shutil.which('simulstream_inference')
        if not simulstream_cmd:
            print("\nError: simulstream_inference not found in PATH.", file=sys.stderr)
            print("Make sure simulstream is installed and in your PATH.", file=sys.stderr)
            return 1

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
        print(f"Audio: {args.manifest or args.wav_list}")
        print(f"Source language: {args.src_lang} → Target: {args.tgt_lang}")
        print(f"Metrics output: {args.metrics_log}")
        print(f"{'='*70}\n")

        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\nError running simulstream: {e}", file=sys.stderr)
        return e.returncode
    finally:
        if temp_wav_list:
            try:
                Path(temp_wav_list).unlink()
                print(f"Cleaned up temporary wav list: {temp_wav_list}")
            except Exception:
                pass


if __name__ == '__main__':
    sys.exit(main())

