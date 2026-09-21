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
Wrapper to run simulstream (https://github.com/hlt-mt/simulstream) inference with NeMo's streaming
ASR/AST configs (see nemo/collections/asr/inference/utils/simulstream_pipeline_adapter.py).

NeMo config files don't have the 'type' field simulstream requires to locate the speech processor
class; this script adds it (and a couple of other simulstream-required fields) on-the-fly.

Requires `pip install simulstream` (not a NeMo dependency).

Usage with a wav list:
    python run_nemo_simulstream.py \\
        --config path/to/cache_aware_rnnt.yaml \\
        --wav-list audio_list.txt \\
        --src-lang ru \\
        --tgt-lang en \\
        --metrics-log metrics.jsonl

Usage with a NeMo manifest:
    python run_nemo_simulstream.py \\
        --config path/to/cache_aware_rnnt.yaml \\
        --manifest data/manifest.json \\
        --src-lang ru \\
        --tgt-lang en \\
        --metrics-log metrics.jsonl

Any additional `key=value` arguments are applied as NeMo config overrides (e.g. `asr.device_id=1`).
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from nemo.collections.asr.inference.utils.simulstream_manifest_utils import load_manifest_audio_paths
from nemo.utils import logging

# ISO 639-1 code -> full name for NeMo's NMT config. Extend as needed.
LANGUAGE_CODES = {
    "bg": "Bulgarian",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "hu": "Hungarian",
    "it": "Italian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mt": "Maltese",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "sv": "Swedish",
    "ru": "Russian",
    "uk": "Ukrainian",
}

# Languages that don't separate words with whitespace, so simulstream latency (and any other
# word-count-based metric) must be computed at the character level instead. Extend as needed.
LATENCY_UNIT_CHAR_LANGUAGES = {"zh", "ja", "ko", "th", "lo", "my", "km"}


def get_language_name(code: str) -> str:
    """Map a language code to its full name for the NeMo NMT config; unknown codes pass through."""
    return LANGUAGE_CODES.get(code, code)


def get_latency_unit(code: str) -> str:
    """Map a language code to its latency unit for simulstream metrics; unknown codes default to 'word'."""
    return "char" if code in LATENCY_UNIT_CHAR_LANGUAGES else "word"


def add_simulstream_fields(
    cfg_path: str | Path,
    metrics_log: str | Path,
    src_lang: str = None,
    tgt_lang: str = None,
    overrides: list = None,
    reference_manifest: str | Path | None = None,
    output_manifest: str | Path | None = None,
) -> str:
    """
    Load a NeMo config and add the fields simulstream requires.

    The simulstream `speech_chunk_size` is always derived from the NeMo config
    (`streaming.chunk_size` for buffered decoding, or `streaming.att_context_size` for cache-aware
    models). The generated config is saved alongside the metrics log.

    Args:
        cfg_path: Path to the NeMo config file.
        metrics_log: Path to the output metrics log file; the generated config is saved in the same
            directory.
        src_lang: Source language code.
        tgt_lang: Target language code.
        overrides: List of "key=value" strings to override config fields.
        reference_manifest: Optional path to a manifest with reference text, for WER calculation.
        output_manifest: Optional path to write a NeMo-style prediction manifest.

    Returns:
        Path to the generated config file with the required fields added.
        This config can be used to later evaluate performance metrics in SimulStream or OmniSTEval.
    """
    cfg = OmegaConf.load(cfg_path)

    if overrides:
        logging.info("Applying command-line overrides:")
        try:
            override_conf = OmegaConf.from_dotlist(overrides)
            cfg = OmegaConf.merge(cfg, override_conf)
            for ov in overrides:
                logging.info(f"  {ov}")
        except Exception as e:
            logging.error(f"  Error applying overrides {overrides}: {e}")

    if src_lang is not None:
        cfg.nmt.source_language = get_language_name(src_lang)
    if tgt_lang is not None:
        cfg.nmt.target_language = get_language_name(tgt_lang)

    if 'type' in cfg:
        logging.info(
            f"Config already has 'type' field, assuming it's already prepared for simulstream and using as-is: {cfg_path}"
        )
        return str(cfg_path)

    logging.info(f"Adding simulstream fields to config: {cfg_path}")

    if 'streaming' in cfg and 'chunk_size' in cfg.streaming:
        speech_chunk_size = cfg.streaming.chunk_size
        logging.info(f"  Using chunk size from config: {speech_chunk_size}s for buffered decoding")
    elif 'streaming' in cfg and 'att_context_size' in cfg.streaming:
        speech_chunk_size = (cfg.streaming.att_context_size[1] + 1) * 0.08
        logging.info(f"  Using chunk size calculated from att_context_size: {speech_chunk_size}s")
    else:
        raise ValueError(f"No chunk_size or att_context_size found in config: {cfg_path}")

    simulstream_fields = OmegaConf.create(
        {
            'type': 'nemo.collections.asr.inference.utils.simulstream_pipeline_adapter.NeMoStreamingPipelineAdapter',
            'speech_chunk_size': speech_chunk_size,
            'detokenizer_type': 'simuleval',
            'latency_unit': get_latency_unit(tgt_lang),
        }
    )
    if output_manifest:
        simulstream_fields.output_manifest_file = str(Path(output_manifest).resolve())
    if reference_manifest:
        simulstream_fields.reference_manifest = str(Path(reference_manifest).resolve())

    # Merge with simulstream fields taking precedence over anything already in the NeMo config.
    cfg = OmegaConf.merge(simulstream_fields, cfg)

    out_dir = Path(metrics_log).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (Path(cfg_path).stem + '_simulstream.yaml')
    with open(out_path, 'w') as f:
        OmegaConf.save(cfg, f)

    # Saved alongside script outputs so it can be used to
    # evaluate performance metrics in SimulStream or OmniSTEval.
    logging.info(f"  Saved config to: {out_path}")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(
        description='Run simulstream inference with a NeMo streaming ASR(+NMT) pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config', required=True, help='Path to NeMo config file (YAML)')

    audio_group = parser.add_mutually_exclusive_group(required=True)
    audio_group.add_argument('--wav-list', help='Path to text file containing audio file paths (one per line)')
    audio_group.add_argument('--manifest', help='Path to NeMo manifest file (JSONL, with audio_filepath field)')

    parser.add_argument('--src-lang', required=True, help='Source language code (e.g., "ru", "en")')
    parser.add_argument('--tgt-lang', required=True, help='Target language code (e.g., "en", "es")')
    parser.add_argument(
        '--metrics-log', default='metrics.jsonl', help='Path to output metrics log file (default: metrics.jsonl)'
    )
    parser.add_argument(
        '--output-manifest',
        default=None,
        help='Path to output prediction manifest JSONL (contains pred_text/pred_translation)',
    )
    args, unknown_args = parser.parse_known_args()

    # Unknown args of the form key=value are passed through as NeMo config overrides.
    overrides = []
    for arg in unknown_args:
        if arg.startswith("--"):
            logging.warning(f"Unknown argument: {arg}")
        elif "=" in arg:
            overrides.append(arg)
        else:
            logging.warning(f"Ignoring unknown argument (expected key=value): {arg}")

    wav_list_path = args.wav_list
    temp_wav_list = None
    try:
        if args.manifest:
            logging.info(f"Loading audio paths from manifest: {args.manifest}")
            audio_paths = load_manifest_audio_paths(args.manifest)
            if not audio_paths:
                raise RuntimeError(f"No audio files found in manifest: {args.manifest}")
            _, temp_wav_list = tempfile.mkstemp(suffix='.txt', prefix='wav_list_')
            with open(temp_wav_list, 'w') as f:
                for path in audio_paths:
                    f.write(f"{path}\n")
            wav_list_path = temp_wav_list
            logging.info(f"Created temporary wav list: {temp_wav_list}")

        config_path = add_simulstream_fields(
            args.config,
            args.metrics_log,
            args.src_lang,
            args.tgt_lang,
            overrides,
            reference_manifest=args.manifest,
            output_manifest=args.output_manifest,
        )

        simulstream_cmd = shutil.which('simulstream_inference')
        if not simulstream_cmd:
            raise RuntimeError(
                "simulstream_inference not found in PATH. "
                "Make sure simulstream is installed (`pip install simulstream`) and in your PATH."
            )

        cmd = [
            simulstream_cmd,
            '--speech-processor-config',
            config_path,
            '--wav-list-file',
            wav_list_path,
            '--src-lang',
            args.src_lang,
            '--tgt-lang',
            args.tgt_lang,
            '--metrics-log-file',
            args.metrics_log,
        ]

        logging.info(
            f"Running simulstream inference: config={args.config}, "
            f"audio={args.manifest or args.wav_list}, "
            f"src_lang={args.src_lang}, tgt_lang={args.tgt_lang}, metrics_log={args.metrics_log}"
        )

        subprocess.run(cmd, check=True)
    finally:
        if temp_wav_list:
            try:
                Path(temp_wav_list).unlink()
                logging.info(f"Cleaned up temporary wav list: {temp_wav_list}")
            except OSError as e:
                logging.warning(f"Failed to clean up temporary wav list {temp_wav_list}: {e}")


if __name__ == '__main__':
    main()
