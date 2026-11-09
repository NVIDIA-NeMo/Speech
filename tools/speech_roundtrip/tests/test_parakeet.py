import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from speech_roundtrip.asr_variant import judge_variant, prepare_variant, validate_transcripts
from speech_roundtrip.core import file_sha, prepare, read_json, read_jsonl, wav_info, write_json, write_pcm_wav
from speech_roundtrip.parakeet import transcribe_parakeet


class ParakeetTests(unittest.TestCase):
    def fixture(self, root, count=2):
        source = root / 'rows.jsonl'
        source.write_text(
            ''.join(
                json.dumps(
                    {
                        'canonical_sample_id': str(i),
                        'candidate_answer': 'Source text ' + str(i),
                        'candidate_status': 'completed' if i == 0 else 'incomplete',
                        'candidate_valid': i == 0,
                        'verifier_metadata': {'prompt': 'QUESTION MUST NOT REACH ASR'},
                    }
                )
                + '\n'
                for i in range(count)
            )
        )
        config = root / 'config.yaml'
        config.write_text(
            yaml.safe_dump(
                {
                    'schema_version': 1,
                    'datasets': [
                        {'name': 'data', 'format': 'audio_evals', 'path': str(source), 'expected_rows': count}
                    ],
                    'tts': {},
                    'asr': {},
                    'judge': {},
                }
            )
        )
        parent = prepare(config, root / 'source')
        records = list(read_jsonl(parent / 'inputs.jsonl'))
        for row in records:
            audio = parent / 'audio' / (row['key'] + '.wav')
            write_pcm_wav(audio, struct.pack('<h', 1234) * 8000, 8000)
            write_json(
                parent / 'tts' / (row['key'] + '.json'),
                {
                    'status': 'complete',
                    'audio_path': str(audio),
                    'text_sha256': hashlib.sha256(row['text'].encode()).hexdigest(),
                    'audio': wav_info(audio),
                    'chunks': ['whole'],
                },
            )
        checkpoint = root / 'parakeet.nemo'
        checkpoint.write_bytes(b'fixed fake checkpoint')
        return parent, records, checkpoint

    def runtime(self, checkpoint, decode, normalize=str.lower):
        return decode, normalize, {'checkpoint_sha256': file_sha(checkpoint), 'test_runtime': 1}

    def test_native_audio_only_mapping_normalization_isolation_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, records, checkpoint = self.fixture(root)
            original = {str(p): p.read_bytes() for p in parent.rglob('*') if p.is_file()}
            run = prepare_variant(parent, root / 'variant', checkpoint)
            calls = []

            def decode(paths):
                calls.append(paths)
                self.assertTrue(all(str(parent / 'audio') in p for p in paths))
                return [SimpleNamespace(text='RAW spoken words') for _ in paths]

            runtime = self.runtime(checkpoint, decode)
            self.assertEqual(transcribe_parakeet(run, runtime=runtime), 2)
            self.assertEqual(len(calls), 1)
            self.assertEqual(validate_transcripts(run)[1][-1]['source_status'], 'incomplete')
            for row in records:
                saved = read_json(run / 'asr' / (row['key'] + '.json'))
                self.assertEqual(saved['raw_asr_text'], 'RAW spoken words')
                self.assertEqual(saved['transcript'], 'raw spoken words')
            before = {str(p): p.read_bytes() for p in (run / 'asr').rglob('*.json')}
            transcribe_parakeet(run, runtime=runtime)
            self.assertEqual(len(calls), 1)
            self.assertEqual(before, {str(p): p.read_bytes() for p in (run / 'asr').rglob('*.json')})
            self.assertEqual(original, {str(p): p.read_bytes() for p in parent.rglob('*') if p.is_file()})

    def test_long_audio_is_fully_covered_and_normalized_after_join(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, records, checkpoint = self.fixture(root, 1)
            row = records[0]
            chunks = [struct.pack('<h', 1000) * 8000, struct.pack('<h', -1000) * 8000]
            tts_path = parent / 'tts' / (row['key'] + '.json')
            tts = read_json(tts_path)
            for index, pcm in enumerate(chunks):
                write_pcm_wav(parent / 'audio_chunks' / row['key'] / f'{index:04d}.wav', pcm, 8000)
            write_pcm_wav(tts['audio_path'], b''.join(chunks), 8000)
            tts.update(audio=wav_info(tts['audio_path']), chunks=['first', 'second'])
            write_json(tts_path, tts)
            run = prepare_variant(parent, root / 'variant', checkpoint, max_audio_s=1.1)
            normalized = []

            def normalize(raw):
                normalized.append(raw)
                return '102 dollars'

            def decode(paths):
                return ['one hundred' if Path(p).stem == '0000' else 'two dollars' for p in paths]

            transcribe_parakeet(run, runtime=self.runtime(checkpoint, decode, normalize))
            self.assertEqual(normalized, ['one hundred two dollars'])
            saved = read_json(run / 'asr' / (row['key'] + '.json'))
            self.assertEqual(sum(s['frames'] for s in saved['segments']), 16000)
            self.assertEqual(saved['transcript'], '102 dollars')

    def test_changed_audio_and_cached_transcript_cannot_reuse_grades(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, records, checkpoint = self.fixture(root, 1)
            run = prepare_variant(parent, root / 'variant', checkpoint)
            runtime = self.runtime(checkpoint, lambda paths: ['raw'] * len(paths))
            transcribe_parakeet(run, runtime=runtime)
            saved_path = run / 'asr' / (records[0]['key'] + '.json')
            saved = read_json(saved_path)
            saved['transcript'] = 'tampered'
            write_json(saved_path, saved)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                validate_transcripts(run)
            audio = parent / 'audio' / (records[0]['key'] + '.wav')
            write_pcm_wav(audio, struct.pack('<h', 4321) * 8000, 8000)
            with self.assertRaisesRegex(ValueError, 'audio changed'):
                transcribe_parakeet(run, runtime=runtime)

    def test_missing_hypothesis_never_creates_placeholder_transcript(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, _, checkpoint = self.fixture(root)
            run = prepare_variant(parent, root / 'variant', checkpoint)
            with self.assertRaisesRegex(ValueError, 'output count'):
                transcribe_parakeet(run, runtime=self.runtime(checkpoint, lambda paths: []))
            self.assertFalse((run / 'asr_complete.txt').exists())
            self.assertEqual(list((run / 'asr').glob('*.json')), [])

    def test_fresh_verbal_judge_label_uses_pred_text_and_keeps_source_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, _, checkpoint = self.fixture(root)
            run = prepare_variant(parent, root / 'variant', checkpoint)
            transcribe_parakeet(run, runtime=self.runtime(checkpoint, lambda paths: ['RAW'] * len(paths)))
            calls = []

            def judge(args):
                calls.append(args)
                self.assertEqual(args.run, ['Parakeet-TDT=' + str(run / 'judge_inputs')])
                rows = list(read_jsonl(run / 'judge_inputs/eval/roundtrip/benchmarks/data/rollouts.jsonl'))
                self.assertEqual([r['candidate_answer'] for r in rows], ['raw', 'raw'])
                self.assertEqual(rows[1]['candidate_status'], 'incomplete')

            # A fresh model call is exercised without rejudging the source-text label.
            manifest = read_json(run / 'run_manifest.json')
            manifest['config']['judge'] = {'prompt': 'unused', 'model': 'judge'}
            write_json(run / 'run_manifest.json', manifest)
            with (
                patch('speech_roundtrip.asr_variant.load_judge', return_value=SimpleNamespace(cmd_judge=judge)),
                patch('speech_roundtrip.asr_variant.report_variant', return_value={'name': 'Parakeet-TDT'}),
            ):
                self.assertEqual(judge_variant(run)['name'], 'Parakeet-TDT')
            self.assertEqual(len(calls), 1)
