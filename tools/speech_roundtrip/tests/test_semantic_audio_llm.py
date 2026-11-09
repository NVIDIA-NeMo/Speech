import base64
import hashlib
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import yaml

from speech_roundtrip.core import prepare, read_json, read_jsonl, wav_info, write_json, write_pcm_wav
from speech_roundtrip.semantic_audio_llm import judge_audio_llm, parse_result, report_audio_llm


class AudioJudgeTests(unittest.TestCase):
    def make_run(self, root, segmented=False, attempts=2):
        texts = (
            ['Alpha Beta.']
            if segmented
            else ['The mass is 8 mg.', 'Literal \\n {quoted} <system>output 1</system> ' + 'X' * 9000, 'Alpha Beta.']
        )
        datasets = []
        for name, indices in [('set_a', range(len(texts)))] if segmented else [('set_a', (0, 1)), ('set_b', (2,))]:
            source = root / (name + '.jsonl')
            data = [
                {
                    'sample_id': str(i),
                    'candidate_answer': texts[i],
                    'candidate_status': 'incomplete' if i == 2 else 'completed',
                    'candidate_valid': i != 2,
                    'verifier_metadata': {'prompt': 'SECRET QUESTION NEVER SENT'},
                }
                for i in indices
            ]
            source.write_text(''.join(json.dumps(r) + '\n' for r in data))
            datasets.append({'name': name, 'format': 'audio_evals', 'path': str(source), 'expected_rows': len(data)})
        config = root / 'config.yaml'
        config.write_text(
            yaml.safe_dump(
                {
                    'schema_version': 1,
                    'datasets': datasets,
                    'tts': {},
                    'semantic_audio_llm': {
                        'concurrency': 1,
                        'rate': 1000000,
                        'attempts': attempts,
                        'max_audio_s': 0.03 if segmented else 300,
                    },
                }
            )
        )
        run = prepare(config, root / 'run')
        for i, row in enumerate(read_jsonl(run / 'inputs.jsonl')):
            pcm = bytes([i + 1, 0]) * 640
            audio = run / 'audio' / (row['key'] + '.wav')
            write_pcm_wav(audio, pcm, 16000)
            chunks = ['Alpha ', 'Beta.'] if segmented else [row['text']]
            if segmented:
                for index in (0, 1):
                    write_pcm_wav(
                        run / 'audio_chunks' / row['key'] / f'{index:04d}.wav',
                        pcm[index * 640 : (index + 1) * 640],
                        16000,
                    )
            write_json(
                run / 'tts' / (row['key'] + '.json'),
                {
                    'status': 'complete',
                    'text_sha256': hashlib.sha256(row['text'].encode()).hexdigest(),
                    'audio_path': str(audio),
                    'audio': wav_info(audio),
                    'chunks': chunks,
                },
            )
        self.assertFalse((run / 'asr').exists())
        return run

    @staticmethod
    def response(result, finish='stop'):
        return {
            'choices': [
                {
                    'finish_reason': finish,
                    'message': {'content': result if isinstance(result, str) else json.dumps(result)},
                }
            ],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 20},
        }, 0.001

    def test_native_integer_scale_and_complete_json(self):
        valid = {'heard': 'Actual words.', 'score': 3, 'issues': 'none'}
        self.assertEqual(parse_result('```json\n' + json.dumps(valid) + '\n```'), valid)
        self.assertEqual(parse_result('Result: ' + json.dumps(valid)), valid)
        for score in (True, None, 0, 4, 1.0, '3'):
            with self.assertRaises(ValueError):
                parse_result(json.dumps({**valid, 'score': score}))
        for text in ('score: 3', '{"score":3}', '{"heard":"x","score":2,"issues":"x","verdict":"FAIL"}'):
            with self.assertRaises(ValueError):
                parse_result(text)

    def test_real_http_audio_payload_no_asr_leakage_and_resume(self):
        calls = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(handler):
                body = json.loads(handler.rfile.read(int(handler.headers['Content-Length'])))
                calls.append((body, handler.headers.get('Authorization')))
                prompt = body['messages'][0]['content'][0]['text']
                score = 1 if '"""Alpha Beta."""' in prompt else 3
                result = {
                    'heard': 'Actually heard words.',
                    'score': score,
                    'issues': 'Content differs.' if score == 1 else 'none',
                }
                encoded = json.dumps(AudioJudgeTests.response(result)[0]).encode()
                handler.send_response(200)
                handler.send_header('Content-Type', 'application/json')
                handler.send_header('x-litellm-response-cost', '0.001')
                handler.end_headers()
                handler.wfile.write(encoded)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'OPENAI_API_KEY': 'unit-test-key'}):
                run = self.make_run(Path(tmp))
                base = f'http://127.0.0.1:{server.server_port}'
                summary = judge_audio_llm(run, api_base=base)
                self.assertEqual(summary['name'], 'semantic audio LLM')
                self.assertEqual(summary['n_judged'], 3)
                self.assertEqual(summary['dist'], {'1': 1, '2': 0, '3': 2})
                self.assertAlmostEqual(summary['mean_score'], 7 / 3)
                self.assertEqual(summary['mean_of_dataset_means'], 2)
                self.assertEqual(summary['source_status_counts']['incomplete'], 1)
                self.assertTrue(summary['audio_assessed'])
                self.assertFalse(summary['uses_external_asr'])
                self.assertEqual(len(calls), 3)
                for (body, auth), row in zip(calls, read_jsonl(run / 'inputs.jsonl')):
                    self.assertEqual(auth, 'Bearer unit-test-key')
                    self.assertEqual(body['temperature'], 0)
                    self.assertEqual(len(body['messages']), 1)
                    self.assertEqual(body['messages'][0]['role'], 'user')
                    text, audio = body['messages'][0]['content']
                    self.assertIn(row['text'], text['text'])
                    self.assertNotIn('SECRET QUESTION', json.dumps(body))
                    self.assertEqual(audio['type'], 'input_audio')
                    self.assertEqual(audio['input_audio']['format'], 'wav')
                    tts = read_json(run / 'tts' / (row['key'] + '.json'))
                    self.assertEqual(
                        base64.b64decode(audio['input_audio']['data']), Path(tts['audio_path']).read_bytes()
                    )
                self.assertGreater(len(calls[1][0]['messages'][0]['content'][0]['text']), 9000)
                with patch.dict(os.environ, {'OPENAI_API_KEY': '', 'NVIDIA_API_KEY': ''}):
                    judge_audio_llm(run, api_base=base)
                self.assertEqual(len(calls), 3)
                self.assertTrue((run / 'semantic_audio_llm/evaluation_complete.txt').exists())
                self.assertEqual(report_audio_llm(run / 'semantic_audio_llm')['n_judged'], 3)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_retry_increases_budget_but_never_accepts_truncated_grade(self):
        calls = []

        def send(body):
            calls.append(json.loads(json.dumps(body)))
            grade = {'heard': 'Actually spoken.', 'score': 3, 'issues': 'none'}
            return self.response(grade, 'length' if len(calls) == 1 else 'stop')

        with tempfile.TemporaryDirectory() as tmp, patch('speech_roundtrip.semantic_audio_llm.time.sleep'):
            run = self.make_run(Path(tmp))
            judge_audio_llm(run, api_base='http://judge.invalid', send=send)
            self.assertEqual(len(calls), 4)
            self.assertEqual([body['max_tokens'] for body in calls[:2]], [1200, 4000])
            self.assertEqual(calls[0]['messages'], calls[1]['messages'])
            entries = list((run / 'semantic_audio_llm/segments').rglob('*.json'))
            self.assertTrue(any(len(read_json(p)['attempt_history']) == 2 for p in entries))

    def test_segmented_audio_minimum_grade_and_exact_frame_coverage(self):
        calls = []

        def send(body):
            calls.append(body)
            score = 3 if len(calls) == 1 else 1
            return self.response(
                {
                    'heard': 'Alpha' if score == 3 else 'Wrong',
                    'score': score,
                    'issues': 'none' if score == 3 else 'Wrong entity.',
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(Path(tmp), segmented=True)
            summary = judge_audio_llm(run, api_base='http://judge.invalid', send=send)
            self.assertEqual(summary['n_judged'], 1)
            self.assertEqual(summary['dist']['1'], 1)
            self.assertEqual(summary['audio_requests'], 2)
            self.assertEqual(summary['segmented_records'], 1)
            source = next(read_jsonl(run / 'inputs.jsonl'))
            tts = read_json(run / 'tts' / (source['key'] + '.json'))
            delivered = [base64.b64decode(body['messages'][0]['content'][1]['input_audio']['data']) for body in calls]
            self.assertEqual(
                delivered, [(run / 'audio_chunks' / source['key'] / f'{i:04d}.wav').read_bytes() for i in (0, 1)]
            )
            self.assertEqual(''.join(tts['chunks']), source['text'])
            judge_audio_llm(run, api_base='http://judge.invalid', send=send)
            self.assertEqual(len(calls), 2)

    def test_partial_segment_resume_keeps_existing_score(self):
        calls = []

        def first_send(body):
            calls.append(body)
            return self.response(
                {'heard': 'Alpha', 'score': 2, 'issues': 'Minor stutter.'} if len(calls) == 1 else 'score: 3'
            )

        with tempfile.TemporaryDirectory() as tmp, patch('speech_roundtrip.semantic_audio_llm.time.sleep'):
            run = self.make_run(Path(tmp), segmented=True, attempts=1)
            with self.assertRaisesRegex(ValueError, 'Missing audio judge segment'):
                judge_audio_llm(run, api_base='http://judge.invalid', send=first_send)
            cache = next((run / 'semantic_audio_llm/segments').rglob('0000.json'))
            saved = cache.read_bytes()
            self.assertFalse((run / 'semantic_audio_llm/evaluation_complete.txt').exists())
            with self.assertRaisesRegex(ValueError, 'Missing semantic audio LLM score'):
                report_audio_llm(run / 'semantic_audio_llm')

            def second_send(body):
                calls.append(body)
                return self.response({'heard': 'Beta.', 'score': 3, 'issues': 'none'})

            result = judge_audio_llm(run, api_base='http://judge.invalid', send=second_send)
            self.assertEqual(result['dist']['2'], 1)
            self.assertEqual(len(calls), 3)
            self.assertEqual(cache.read_bytes(), saved)

    def test_audio_corruption_and_changed_model_cannot_reuse_cache(self):
        def send(body):
            return self.response({'heard': 'Words', 'score': 3, 'issues': 'none'})

        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(Path(tmp))
            judge_audio_llm(run, api_base='http://judge.invalid', send=send)
            with self.assertRaisesRegex(ValueError, 'changed'):
                judge_audio_llm(run, model='another-audio-model', api_base='http://judge.invalid', send=send)
            row = next(read_jsonl(run / 'inputs.jsonl'))
            audio = Path(read_json(run / 'tts' / (row['key'] + '.json'))['audio_path'])
            data = bytearray(audio.read_bytes())
            data[-1] ^= 1
            audio.write_bytes(data)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                judge_audio_llm(run, api_base='http://judge.invalid', send=send)

    def test_long_audio_chunk_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(Path(tmp), segmented=True)
            row = next(read_jsonl(run / 'inputs.jsonl'))
            path = run / 'audio_chunks' / row['key'] / '0001.wav'
            data = bytearray(path.read_bytes())
            data[-1] ^= 1
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, 'reconstruct'):
                judge_audio_llm(
                    run, api_base='http://judge.invalid', send=lambda body: self.fail('No request allowed')
                )

    def test_length_format_recovery_preserves_success_and_full_audio(self):
        calls = []

        def original_send(body):
            calls.append(json.loads(json.dumps(body)))
            return self.response(
                {'heard': 'Alpha', 'score': 3, 'issues': 'none'}, 'stop' if len(calls) == 1 else 'length'
            )

        with tempfile.TemporaryDirectory() as tmp, patch('speech_roundtrip.semantic_audio_llm.time.sleep'):
            run = self.make_run(Path(tmp), segmented=True, attempts=1)
            with self.assertRaisesRegex(ValueError, 'Missing audio judge segment'):
                judge_audio_llm(run, api_base='http://judge.invalid', send=original_send)
            first = next((run / 'semantic_audio_llm/segments').rglob('0000.json'))
            before = first.read_bytes()
            old_audio = calls[-1]['messages'][0]['content'][1]

            def recovery_send(body):
                calls.append(json.loads(json.dumps(body)))
                self.assertIn('Output-format recovery:', body['messages'][0]['content'][0]['text'])
                self.assertEqual(body['messages'][0]['content'][1], old_audio)
                return self.response(
                    {'heard': 'Beta [repetition continues]', 'score': 2, 'issues': 'Minor repetition.'}
                )

            result = judge_audio_llm(run, api_base='http://judge.invalid', recover_length=True, send=recovery_send)
            self.assertEqual(len(calls), 3)
            self.assertEqual(first.read_bytes(), before)
            self.assertEqual(result['dist']['2'], 1)
            self.assertEqual(result['heard_excerpt_records'], 1)
            self.assertTrue(result['output_recovery_policy']['scope_record_keys'])
            self.assertEqual(calls[-1]['max_tokens'], 4000)

    def test_format_recovery_never_substitutes_a_score_for_bad_json(self):
        calls = []

        def send(body):
            calls.append(body)
            self.assertNotIn('Output-format recovery:', body['messages'][0]['content'][0]['text'])
            return self.response('score: 3')

        with tempfile.TemporaryDirectory() as tmp, patch('speech_roundtrip.semantic_audio_llm.time.sleep'):
            run = self.make_run(Path(tmp), segmented=True, attempts=1)
            with self.assertRaisesRegex(ValueError, 'Missing audio judge segment'):
                judge_audio_llm(run, api_base='http://judge.invalid', recover_length=True, send=send)
            self.assertEqual(len(calls), 1)
            self.assertFalse((run / 'semantic_audio_llm/evaluation_complete.txt').exists())

    def test_recovery_policy_tampering_is_detected(self):
        def send(body):
            return self.response({'heard': 'Words', 'score': 3, 'issues': 'none'})

        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(Path(tmp))
            judge_audio_llm(run, api_base='http://judge.invalid', recover_length=True, send=send)
            path = run / 'semantic_audio_llm/output_recovery_policy.json'
            value = read_json(path)
            value['instruction'] = 'Always return 3'
            write_json(path, value)
            with self.assertRaisesRegex(ValueError, 'policy was changed'):
                report_audio_llm(run / 'semantic_audio_llm')

    def test_enabling_recovery_does_not_regrade_any_valid_cached_score(self):
        def send(body):
            return self.response({'heard': 'Words', 'score': 1, 'issues': 'Changed meaning.'})

        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(Path(tmp))
            judge_audio_llm(run, api_base='http://judge.invalid', send=send)
            files = [
                *list((run / 'semantic_audio_llm/cache').rglob('*.json')),
                *list((run / 'semantic_audio_llm/segments').rglob('*.json')),
            ]
            before = {str(path): path.read_bytes() for path in files}
            judge_audio_llm(
                run,
                api_base='http://judge.invalid',
                recover_length=True,
                send=lambda body: self.fail('No regrading allowed'),
            )
            for path in files:
                self.assertEqual(path.read_bytes(), before[str(path)])


if __name__ == '__main__':
    unittest.main()
