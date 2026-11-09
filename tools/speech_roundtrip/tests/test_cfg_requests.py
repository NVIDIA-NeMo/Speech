import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from speech_roundtrip.core import prepare, read_json, read_jsonl
from speech_roundtrip.tts import synthesize


class CFGRequestTests(unittest.TestCase):
    def test_cfg_parameters_and_silent_model_failure_are_preserved(self):
        """A valid silent PCM reply is a measured failure, not a request to resample."""
        calls = []

        def response(request, **kwargs):
            calls.append(json.loads(request.data))
            result = io.BytesIO(b'\0\0' * 2205)
            result.headers = {'Content-Type': 'audio/pcm'}
            return result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.jsonl'
            source.write_text(json.dumps({'id': 'test', 'question': 'Question', 'text': 'Say this.'}) + '\n')
            config = root / 'config.yaml'
            config.write_text(
                yaml.safe_dump(
                    {
                        'schema_version': 1,
                        'datasets': [{'name': 'fixture', 'path': str(source), 'expected_rows': 1}],
                        'tts': {
                            'url': 'http://unused',
                            'voice': 'voice',
                            'sample_rate': 22050,
                            'request_options': {'extra_params': {'temperature': 0.7, 'top_k': 80, 'cfg_scale': 2.5}},
                        },
                        'asr': {},
                        'judge': {},
                    }
                )
            )
            run = prepare(config, root / 'run')
            with (
                patch('speech_roundtrip.tts.wait_ready'),
                patch('speech_roundtrip.tts.urllib.request.urlopen', side_effect=response),
            ):
                synthesize(run)
                synthesize(run)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]['extra_params'], {'temperature': 0.7, 'top_k': 80, 'cfg_scale': 2.5})
            row = next(read_jsonl(run / 'inputs.jsonl'))
            saved = read_json(run / 'tts' / (row['key'] + '.json'))
            self.assertEqual(saved['status'], 'complete')
            self.assertEqual(saved['audio']['rms'], 0)
            self.assertTrue(saved['silent_audio'])
            self.assertEqual(saved['silent_chunks'], [0])
            self.assertTrue(saved['single_turn'])


if __name__ == '__main__':
    unittest.main()
