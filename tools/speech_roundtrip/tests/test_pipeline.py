import json
import math
import struct
import tempfile
import threading
import time
import unittest
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import yaml
from speech_roundtrip.asr import ASRResponseError, _decode_audio, transcribe
from speech_roundtrip.core import (
    adapt_dataset,
    chunk_text,
    prepare,
    read_json,
    read_jsonl,
    run_records,
    wav_info,
    write_json,
    write_pcm_wav,
)
from speech_roundtrip.tts import synthesize


class PipelineTests(unittest.TestCase):
    def test_empty_api_content_is_not_confused_with_reasoning_or_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'audio.wav'
            write_pcm_wav(path, struct.pack('<h', 1234) * 2205, 22050)
            response = {'choices': [{'finish_reason': 'stop', 'message': {'content': None}}]}
            with patch('speech_roundtrip.asr.request_json', return_value=response):
                raw, text = _decode_audio('http://unused', {'model': 'asr', 'prompt': 'Transcribe only.'}, path)
                self.assertEqual(text, '')
                self.assertIs(raw, response)
            for extra in (
                {'reasoning_content': 'Not a transcript'},
                {'refusal': 'Refused'},
                {'tool_calls': [{'name': 'unexpected_tool'}]},
            ):
                response = {'choices': [{'finish_reason': 'stop', 'message': {'content': None, **extra}}]}
                with patch('speech_roundtrip.asr.request_json', return_value=response):
                    with self.assertRaises(ASRResponseError) as raised:
                        _decode_audio('http://unused', {'model': 'asr', 'prompt': 'Transcribe only.'}, path)
                    self.assertIs(raised.exception.response, response)

    def test_asr_length_recovery_covers_all_audio_and_resumes_segments(self):
        import base64
        import io

        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"data":[]}')

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(payload)
                first = len(requests) == 1
                text = (
                    'TRUNCATED TEXT MUST NOT BE USED'
                    if first
                    else (None if len(requests) == 2 else ('' if len(requests) == 3 else 'Recovered.'))
                )
                self.send_response(200)
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {'choices': [{'finish_reason': 'length' if first else 'stop', 'message': {'content': text}}]}
                    ).encode()
                )

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / 'source.jsonl'
                source.write_text(
                    json.dumps({'id': '1', 'question': 'SECRET QUESTION', 'text': 'SECRET SOURCE'}) + '\n'
                )
                cfg = root / 'config.yaml'
                cfg.write_text(
                    yaml.safe_dump(
                        {
                            'schema_version': 1,
                            'datasets': [{'name': 'new_dataset', 'path': str(source), 'expected_rows': 1}],
                            'tts': {},
                            'asr': {
                                'url': f'http://127.0.0.1:{server.server_port}',
                                'model': 'asr',
                                'prompt': 'Transcribe only.',
                                'recovery_max_audio_s': 1,
                                'minimum_recovery_audio_s': 0.2,
                            },
                            'judge': {},
                        }
                    )
                )
                run = prepare(cfg, root / 'run')
                row = next(read_jsonl(run / 'inputs.jsonl'))
                pcm = struct.pack('<h', 3000) * 33075 + b'\0\0' * 4410 + struct.pack('<h', -3000) * 33075
                audio = run / 'audio' / f"{row['key']}.wav"
                write_pcm_wav(audio, pcm, 22050)
                write_json(
                    run / 'tts' / f"{row['key']}.json",
                    {'status': 'complete', 'audio_path': str(audio), 'audio': wav_info(audio), 'chunks': ['source']},
                )
                transcribe(run)
                path = run / 'asr' / f"{row['key']}.json"
                result = read_json(path)
                self.assertNotIn('TRUNCATED', result['transcript'])
                self.assertEqual(result['segmentation'], 'adaptive_audio_ranges')
                self.assertTrue(any(s['transcript'] == '' for s in result['segments']))
                forwarded = []
                for payload in requests[1:]:
                    self.assertNotIn('SECRET', json.dumps(payload))
                    url = payload['messages'][0]['content'][1]['audio_url']['url']
                    with wave.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1])), 'rb') as w:
                        self.assertLessEqual(w.getnframes() / w.getframerate(), 1)
                        forwarded.append(w.readframes(w.getnframes()))
                self.assertEqual(b''.join(forwarded), pcm)
                calls = len(requests)
                path.unlink()  # Simulate loss of the reply-level commit, keeping segment checkpoints.
                transcribe(run)
                self.assertEqual(len(requests), calls)
                self.assertEqual(read_json(path)['transcript'], result['transcript'])
        finally:
            server.shutdown()
            server.server_close()

    def test_checksum_validation_keeps_wav_integrity_without_recomputing_rms(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'audio.wav'
            write_pcm_wav(path, struct.pack('<h', 1234) * 2205, 22050)
            complete = wav_info(path)
            fast = wav_info(path, include_rms=False)
            self.assertEqual(fast, {k: v for k, v in complete.items() if k != 'rms'})
            path.write_bytes(path.read_bytes()[:-2])
            with self.assertRaisesRegex(ValueError, 'Truncated'):
                wav_info(path, include_rms=False)

    def test_long_asr_preserves_every_frame_and_detects_changed_chunks(self):
        import base64
        import io

        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"data":[]}')

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(payload)
                text = 'First part.' if len(requests) == 1 else 'Second part.'
                self.send_response(200)
                self.end_headers()
                self.wfile.write(
                    json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': text}}]}).encode()
                )

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / 'source.jsonl'
                source.write_text(
                    json.dumps({'id': '1', 'question': 'SECRET QUESTION', 'text': 'SECRET ORIGINAL TEXT'}) + '\n'
                )
                cfg = root / 'config.yaml'
                cfg.write_text(
                    yaml.safe_dump(
                        {
                            'schema_version': 1,
                            'datasets': [{'name': 'new_dataset', 'path': str(source), 'expected_rows': 1}],
                            'tts': {},
                            'asr': {
                                'url': f'http://127.0.0.1:{server.server_port}',
                                'model': 'asr',
                                'prompt': 'Transcribe only.',
                                'max_single_request_audio_s': 0.15,
                            },
                            'judge': {},
                        }
                    )
                )
                run = prepare(cfg, root / 'run')
                row = next(read_jsonl(run / 'inputs.jsonl'))
                pcm = [struct.pack('<h', 3000) * 2205, struct.pack('<h', -3000) * 2205]
                for i, data in enumerate(pcm):
                    write_pcm_wav(run / 'audio_chunks' / row['key'] / f'{i:04d}.wav', data, 22050)
                audio = run / 'audio' / f"{row['key']}.wav"
                write_pcm_wav(audio, b''.join(pcm), 22050)
                write_json(
                    run / 'tts' / f"{row['key']}.json",
                    {
                        'status': 'complete',
                        'audio_path': str(audio),
                        'audio': wav_info(audio),
                        'chunks': ['part 1', 'part 2'],
                    },
                )
                transcribe(run)
                transcribe(run)
                self.assertEqual(len(requests), 2)
                recovered = []
                for payload in requests:
                    self.assertNotIn('SECRET', json.dumps(payload))
                    url = payload['messages'][0]['content'][1]['audio_url']['url']
                    with wave.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1])), 'rb') as w:
                        recovered.append(w.readframes(w.getnframes()))
                self.assertEqual(b''.join(recovered), b''.join(pcm))
                result = read_json(run / 'asr' / f"{row['key']}.json")
                self.assertEqual(result['transcript'], 'First part. Second part.')
                self.assertEqual(result['segmentation'], 'tts_chunk_boundaries')
                self.assertEqual(len(result['segments']), 2)
                write_pcm_wav(run / 'audio_chunks' / row['key'] / '0001.wav', pcm[0], 22050)
                with self.assertRaisesRegex(ValueError, 'exactly reconstruct'):
                    transcribe(run)
                self.assertEqual(len(requests), 2)
        finally:
            server.shutdown()
            server.server_close()

    def test_bounded_pool_loses_no_records(self):
        active, peak = 0, 0
        seen = set()
        lock = threading.Lock()

        def process(record):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.005 * (record % 3 + 1))
            with lock:
                seen.add(record)
                active -= 1
            return 1

        self.assertEqual(run_records(iter(range(25)), process, 3), 25)
        self.assertLessEqual(peak, 3)
        self.assertGreater(peak, 1)
        self.assertEqual(seen, set(range(25)))

    def test_chunking_preserves_all_characters(self):
        text = 'Hello, world!  Two spaces stay.\nUnicode: café, 日本語. ' * 20
        chunks = chunk_text(text, 80)
        self.assertEqual(''.join(chunks), text)
        self.assertTrue(all(len(c) <= 80 for c in chunks))
        with self.assertRaises(ValueError):
            chunk_text('x' * 100, 40)

    def test_adapter_preserves_failed_source_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'inputs.jsonl'
            path.write_text(
                json.dumps(
                    {
                        'canonical_sample_id': 'id',
                        'candidate_answer': '',
                        'candidate_status': 'incomplete',
                        'candidate_valid': False,
                        'verifier_metadata': {'prompt': 'Question'},
                    }
                )
                + '\n'
            )
            row = next(adapt_dataset({'name': 'new_dataset', 'path': str(path), 'format': 'audio_evals'}))
            self.assertFalse(row['source_valid'])
            self.assertEqual(row['source_status'], 'incomplete')
            self.assertEqual(row['text'], '')

    def test_duplicate_ids_and_changed_inputs_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.jsonl'
            row = {'id': '1', 'question': 'Question', 'text': 'Answer.'}
            source.write_text((json.dumps(row) + '\n') * 2)
            with self.assertRaises(ValueError):
                list(adapt_dataset({'name': 'custom', 'path': str(source)}))
            source.write_text(json.dumps(row) + '\n')
            cfg = root / 'config.yaml'
            cfg.write_text(
                yaml.safe_dump(
                    {
                        'schema_version': 1,
                        'datasets': [{'name': 'custom', 'path': str(source)}],
                        'tts': {},
                        'asr': {},
                        'judge': {},
                    }
                )
            )
            prepare(cfg, root / 'run')
            prepare(cfg, root / 'run')  # Idempotent when inputs/configuration match.
            row['text'] = 'Changed answer.'
            source.write_text(json.dumps(row) + '\n')
            with self.assertRaises(ValueError):
                prepare(cfg, root / 'run')

    def test_real_http_stages_resume_and_repair_corrupt_audio(self):
        requests = []
        pcm = b''.join(struct.pack('<h', int(4000 * math.sin(i / 8))) for i in range(2205))

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"data":[]}')

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append((self.path, payload))
                self.send_response(200)
                self.end_headers()
                if self.path.endswith('/audio/speech'):
                    self.wfile.write(pcm)
                else:
                    self.wfile.write(
                        json.dumps(
                            {'choices': [{'finish_reason': 'stop', 'message': {'content': 'Recovered spoken reply.'}}]}
                        ).encode()
                    )

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / 'source.jsonl'
                source.write_text(
                    json.dumps({'id': '1', 'question': 'SECRET QUESTION', 'text': 'SECRET ORIGINAL ANSWER.'}) + '\n'
                )
                url = f'http://127.0.0.1:{server.server_port}'
                cfg = root / 'config.yaml'
                cfg.write_text(
                    yaml.safe_dump(
                        {
                            'schema_version': 1,
                            'datasets': [{'name': 'unseen_dataset', 'path': str(source), 'expected_rows': 1}],
                            'tts': {'url': url, 'voice': 'reference', 'sample_rate': 22050},
                            'asr': {'url': url, 'model': 'asr', 'prompt': 'Transcribe only.'},
                            'judge': {},
                        }
                    )
                )
                run = prepare(cfg, root / 'run')
                synthesize(run)
                synthesize(run)
                self.assertEqual(len(requests), 1)
                record = next(read_jsonl(run / 'inputs.jsonl'))
                audio = run / 'audio' / f"{record['key']}.wav"
                self.assertEqual(wav_info(audio)['sample_rate'], 22050)
                audio.write_bytes(b'corrupt')
                # Corruption must be detected; it cannot silently count as a reusable result.
                with self.assertRaises(Exception):
                    synthesize(run)
                audio.unlink()
                synthesize(run)
                transcribe(run)
                transcribe(run)
                self.assertEqual(len(requests), 3)
                asr_payload = requests[-1][1]
                serialized = json.dumps(asr_payload)
                self.assertNotIn('SECRET QUESTION', serialized)
                self.assertNotIn('SECRET ORIGINAL ANSWER', serialized)
                self.assertFalse(asr_payload['chat_template_kwargs']['enable_thinking'])
                result = read_json(run / 'asr' / f"{record['key']}.json")
                self.assertEqual(result['transcript'], 'Recovered spoken reply.')
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
