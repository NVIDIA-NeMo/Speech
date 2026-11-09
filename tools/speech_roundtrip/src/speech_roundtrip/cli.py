"""Independently runnable prepare, synthesize, transcribe, judge, and report stages."""

import argparse
import json
from pathlib import Path

from .core import prepare, read_json, read_jsonl, wav_info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--config', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--preview-per-dataset', type=int, default=0)
    for command in ('synthesize', 'transcribe', 'judge', 'report', 'status'):
        p = commands.add_parser(command)
        p.add_argument('--run-dir', required=True)
        if command in ('synthesize', 'transcribe'):
            p.add_argument('--limit', type=int, default=0)
    for command in ('semantic-judge', 'semantic-report'):
        p = commands.add_parser(command)
        p.add_argument('--run-dir', required=True)
        p.add_argument('--output-dir')
        if command == 'semantic-judge':
            p.add_argument('--prompt')
            p.add_argument('--model')
            p.add_argument('--workers', type=int)
    for command in ('semantic-audio-llm', 'semantic-audio-llm-report'):
        p = commands.add_parser(command)
        p.add_argument('--run-dir', required=True)
        p.add_argument('--output-dir')
        if command == 'semantic-audio-llm':
            p.add_argument('--model')
            p.add_argument('--api-base')
            p.add_argument('--concurrency', type=int)
            p.add_argument(
                '--recover-length',
                action='store_true',
                help='After exhausted length failures, allow a marked compact heard excerpt for repetition runs',
            )
            selection = p.add_mutually_exclusive_group()
            selection.add_argument('--limit', type=int, default=0)
            selection.add_argument('--keys', nargs='+')
    p = commands.add_parser('prepare-asr-variant')
    p.add_argument('--source-run', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--name', default='Parakeet-TDT')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--max-audio-s', type=float, default=180)
    for command in ('judge-asr-variant', 'report-asr-variant'):
        p = commands.add_parser(command)
        p.add_argument('--run-dir', required=True)
    args = parser.parse_args(argv)
    if getattr(args, 'limit', 0) < 0:
        parser.error('--limit must be nonnegative')
    if args.command == 'prepare':
        if args.preview_per_dataset < 0:
            parser.error('--preview-per-dataset must be nonnegative')
        run = prepare(args.config, args.run_dir, args.preview_per_dataset)
        print(json.dumps(read_json(run / 'run_manifest.json'), indent=2))
    elif args.command == 'synthesize':
        from .tts import synthesize

        synthesize(args.run_dir, args.limit)
    elif args.command == 'transcribe':
        if read_json(Path(args.run_dir) / 'run_manifest.json')['config']['asr'].get('backend') == 'parakeet_tdt':
            from .parakeet import transcribe_parakeet

            transcribe_parakeet(args.run_dir, args.limit)
        else:
            from .asr import transcribe

            transcribe(args.run_dir, args.limit)
    elif args.command == 'judge':
        from .judging import judge

        judge(args.run_dir)
    elif args.command == 'report':
        from .judging import report

        report(args.run_dir)
    elif args.command == 'semantic-judge':
        from .semantic import judge_semantic

        judge_semantic(args.run_dir, args.prompt, args.model, args.workers, args.output_dir)
    elif args.command == 'semantic-report':
        from .semantic import report_semantic

        report_semantic(args.output_dir or Path(args.run_dir) / 'semantic_judge')
    elif args.command == 'semantic-audio-llm':
        from .semantic_audio_llm import judge_audio_llm

        judge_audio_llm(
            args.run_dir,
            args.model,
            args.concurrency,
            args.output_dir,
            args.api_base,
            args.limit,
            args.keys,
            args.recover_length,
        )
    elif args.command == 'semantic-audio-llm-report':
        from .semantic_audio_llm import report_audio_llm

        report_audio_llm(args.output_dir or Path(args.run_dir) / 'semantic_audio_llm')
    elif args.command == 'prepare-asr-variant':
        from .asr_variant import prepare_variant

        print(
            prepare_variant(
                args.source_run, args.run_dir, args.checkpoint, args.name, args.batch_size, args.max_audio_s
            )
        )
    elif args.command == 'judge-asr-variant':
        from .asr_variant import judge_variant

        judge_variant(args.run_dir)
    elif args.command == 'report-asr-variant':
        from .asr_variant import report_variant

        report_variant(args.run_dir)
    else:
        run = Path(args.run_dir)
        summary = {'scope': read_json(run / 'run_manifest.json')['scope'], 'tts_complete': 0, 'asr_complete': 0}
        for record in read_jsonl(run / 'inputs.jsonl'):
            for stage in ('tts', 'asr'):
                path = run / stage / f"{record['key']}.json"
                if path.exists() and read_json(path).get('status') == 'complete':
                    if stage == 'tts':
                        info = wav_info(read_json(path)['audio_path'], include_rms=False)
                        if info['sha256'] != read_json(path)['audio']['sha256']:
                            raise ValueError('Audio checksum mismatch')
                    summary[stage + '_complete'] += 1
        print(json.dumps(summary, indent=2))
