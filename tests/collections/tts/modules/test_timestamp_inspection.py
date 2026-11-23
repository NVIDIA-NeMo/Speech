# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable report tests that run without importing NeMo or a model runtime."""

import base64
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
import wave
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[4] / 'examples/tts/inspect_nemotron_timestamp_aligner.py'
SPEC = importlib.util.spec_from_file_location('timestamp_inspection', SCRIPT)
inspection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inspection)


class TestTimestampInspection(unittest.TestCase):
    def setUp(self):
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b'\0\0' * 16000)
        self.audio = base64.b64encode(buffer.getvalue()).decode('ascii')

    def _payload(self, html):
        embedded = html.split('<script id="report-data" type="application/json">', 1)[1].split('</script>', 1)[0]
        return json.loads(embedded)

    def test_portable_report_escapes_untrusted_transcripts(self):
        injected = '</script><script>alert("unexpected")</script>'
        record = {
            'id': injected,
            'text': 'Olá, ¿qué tal? ' + injected,
            'language': 'pt',
            'words': [{'word': injected, 'start': 0.1, 'end': 0.4}],
            'audio_wav_base64': self.audio,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = inspection.build_inspection_html([record], Path(directory) / 'report.html')
            html = path.read_text(encoding='utf-8')
        payload = self._payload(html)
        self.assertEqual(payload['records'][0]['text'], record['text'])
        self.assertNotIn(injected, html)
        self.assertEqual(html.count(self.audio), 1)
        self.assertNotIn('https://', html)
        self.assertIn('source.start(0,offset,duration)', html)

    def test_invalid_intervals_are_bounded_and_diagnosed(self):
        record = {
            'audio_wav_base64': self.audio,
            'words': [
                {'word': 'outside', 'start': -0.2, 'end': 1.4},
                {'word': 'reversed', 'start': 0.9, 'end': 0.1},
                {'word': 'nan', 'start': float('nan'), 'end': 0.5},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = inspection.build_inspection_html([record], Path(directory) / 'report.html')
            result = self._payload(path.read_text(encoding='utf-8'))['records'][0]
        self.assertEqual(result['duration'], 1.0)
        self.assertEqual((result['words'][0]['start'], result['words'][0]['end']), (0.0, 1.0))
        self.assertEqual(result['words'][1]['start'], result['words'][1]['end'])
        self.assertEqual(result['words'][2]['start'], result['words'][2]['end'])
        self.assertEqual(len(result['viewer_warnings']), 3)
        self.assertTrue(all(word['interval_issue'] for word in result['words']))

    def test_render_only_cli_preserves_failure_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / 'alignment.json'
            target = directory / 'report.html'
            source.write_text(
                json.dumps(
                    {
                        'metadata': {'model_load_seconds': 3.5},
                        'records': [{'id': 'failed', 'language': 'es', 'text': 'Hola', 'error': 'no alignment'}],
                    }
                ),
                encoding='utf-8',
            )
            subprocess.run(
                [sys.executable, str(SCRIPT), '--alignments-json', str(source), '--output-html', str(target)],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = self._payload(target.read_text(encoding='utf-8'))
        self.assertEqual(payload['metadata']['model_load_seconds'], 3.5)
        self.assertEqual(payload['records'][0]['error'], 'no alignment')
        self.assertEqual(payload['records'][0]['words'], [])

    def test_inference_cli_uses_checkpoint_dtype_and_reports_partial_failure(self):
        records = [
            {
                'id': 'aligned',
                'language': 'en',
                'text': 'Hello',
                'audio_wav_base64': self.audio,
                'words': [{'word': 'Hello', 'start': 0.1, 'end': 0.4}],
                'error': None,
            },
            {
                'id': 'failed',
                'language': 'pt',
                'text': 'Olá',
                'audio_wav_base64': self.audio,
                'words': [],
                'error': 'RuntimeError: teacher execution failed',
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source, target, output = directory / 'manifest.jsonl', directory / 'report.html', directory / 'report.json'
            source.write_text(''.join(json.dumps(record) + '\n' for record in records), encoding='utf-8')
            argv = [
                str(SCRIPT),
                '--manifest',
                str(source),
                '--model-path',
                str(directory / 'model'),
                '--output-html',
                str(target),
                '--output-json',
                str(output),
            ]
            stderr = io.StringIO()
            with patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                with patch.object(inspection, '_align_manifest', return_value=(records, {'errors': 1})) as align:
                    status = inspection.main()
            self.assertEqual(status, 1)
            self.assertEqual(align.call_args.args[1].encoder_dtype, 'checkpoint')
            self.assertTrue(target.is_file())
            payload = self._payload(target.read_text(encoding='utf-8'))
            exported = json.loads(output.read_text(encoding='utf-8'))
        self.assertIn('1/2 recordings', stderr.getvalue())
        self.assertEqual(payload['records'][1]['error'], records[1]['error'])
        self.assertEqual(exported['metadata']['errors'], 1)
        self.assertEqual(len(exported['records'][0]['words']), 1)


if __name__ == '__main__':
    unittest.main()
