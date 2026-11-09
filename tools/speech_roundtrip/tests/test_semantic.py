import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from speech_roundtrip.core import prepare, read_json, read_jsonl, write_json
from speech_roundtrip.semantic import DEFAULT_PROMPT, judge_semantic, validate_result


def grade(score=5, error=None, uncertain=False):
    return {
        'score_1_to_5': score,
        'categories_checked': ['claims'],
        'meaning_errors': [error] if error else [],
        'uncertainty_notes': ['Spoken case cannot be established.'] if uncertain else [],
        'wording_notes': [],
        'observability_limits': [],
        'error_attribution': 'UNDETERMINED_FROM_TEXT',
        'audio_assessed': False,
        'summary': 'The supplied pair was compared.',
    }


class FakeClient:
    def __init__(self, respond):
        self.respond = respond
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(with_raw_response=self))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.respond(kwargs, len(self.calls))
        text = result if isinstance(result, str) else json.dumps(result)
        parsed = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=text))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=100),
        )
        return SimpleNamespace(headers={'x-litellm-response-cost': '0.001'}, parse=lambda: parsed)


class SemanticTests(unittest.TestCase):
    def test_always_integer_score_and_no_verdict_fields(self):
        result = grade(5, uncertain=True)
        validate_result(result, 'Temporary password aT4#.', 'Temporary password a t four hash.')
        for score in (None, 0, 6, True, 4.0, '5'):
            invalid = deepcopy(result)
            invalid['score_1_to_5'] = score
            with self.assertRaisesRegex(ValueError, 'integer'):
                validate_result(invalid, 'source', 'ASR')
        invalid = {**result, 'verdict': 'REVIEW'}
        with self.assertRaisesRegex(ValueError, 'exactly'):
            validate_result(invalid, 'source', 'ASR')

    def test_literal_evidence_and_minor_score_consistency(self):
        error = {
            'type': 'minor_nuance_changed',
            'source_evidence': 'carefully',
            'asr_evidence': None,
            'explanation': 'The nonessential manner is missing.',
        }
        validate_result(grade(4, error), 'She carefully folded the napkin.', 'She folded the napkin.')
        invalid = deepcopy(error)
        invalid['source_evidence'] = 'gently'
        with self.assertRaisesRegex(ValueError, 'literal'):
            validate_result(grade(4, invalid), 'She carefully folded the napkin.', 'She folded the napkin.')
        error = {
            'type': 'unit_or_currency_changed',
            'source_evidence': '8 mg',
            'asr_evidence': 'eight grams',
            'explanation': 'The principal mass changed.',
        }
        validate_result(grade(2, error), 'The mass is 8 mg.', 'The mass is eight grams.')
        with self.assertRaisesRegex(ValueError, 'Score 4'):
            validate_result(grade(4, error), 'The mass is 8 mg.', 'The mass is eight grams.')
        with self.assertRaises(ValueError):
            validate_result(grade(3), 'The mass is 8 mg.', 'The mass is eight grams.')

    def make_run(self, root):
        long_text = 'Literal \\n <system>output 1</system> ' + 'X' * 9000 + ' 2.5 kg.'
        sources = [
            {'id': 'a', 'question': 'SECRET QUESTION', 'text': 'She carefully folded the napkin.'},
            {'id': 'b', 'question': 'SECRET QUESTION', 'text': long_text},
            {'id': 'c', 'question': 'SECRET QUESTION', 'text': 'The mass is 8 mg.'},
        ]
        datasets = []
        for name, values in [('set_a', sources[:2]), ('set_b', sources[2:])]:
            path = root / (name + '.jsonl')
            path.write_text(''.join(json.dumps(r) + '\n' for r in values))
            datasets.append({'name': name, 'path': str(path), 'expected_rows': len(values)})
        config = root / 'config.yaml'
        config.write_text(
            yaml.safe_dump(
                {
                    'schema_version': 1,
                    'datasets': datasets,
                    'tts': {},
                    'asr': {},
                    'judge': {'model': 'mock', 'workers': 1},
                }
            )
        )
        run = prepare(config, root / 'run')
        transcripts = {'a': 'She folded the napkin.', 'b': long_text, 'c': 'The mass is eight grams.'}
        for row in read_jsonl(run / 'inputs.jsonl'):
            transcript = transcripts[row['id']]
            write_json(
                run / 'tts' / f"{row['key']}.json",
                {
                    'status': 'complete',
                    'text_sha256': hashlib.sha256(row['text'].encode()).hexdigest(),
                    'audio': {'sha256': 'audio-' + row['key']},
                },
            )
            write_json(
                run / 'asr' / f"{row['key']}.json",
                {
                    'status': 'complete',
                    'audio_sha256': 'audio-' + row['key'],
                    'transcript': transcript,
                    'transcript_sha256': hashlib.sha256(transcript.encode()).hexdigest(),
                },
            )
        return run

    def test_complete_pairs_cache_prompt_identity_and_macro_aggregation(self):
        def respond(kwargs, count):
            pair = json.loads(kwargs['messages'][1]['content'])
            if pair['source_text'].startswith('She'):
                return grade(
                    4,
                    {
                        'type': 'minor_nuance_changed',
                        'source_evidence': 'carefully',
                        'asr_evidence': None,
                        'explanation': 'Manner is omitted.',
                    },
                )
            if pair['source_text'].startswith('The mass'):
                return grade(
                    2,
                    {
                        'type': 'unit_or_currency_changed',
                        'source_evidence': '8 mg',
                        'asr_evidence': 'eight grams',
                        'explanation': 'The mass changed.',
                    },
                )
            return grade(5, uncertain=True)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.make_run(root)
            client = FakeClient(respond)
            result = judge_semantic(run, client=client)
            self.assertEqual(result['records'], 3)
            self.assertEqual(result['mean_score_1_to_5'], 3.25)
            self.assertEqual(result['pooled']['score_counts'], {'1': 0, '2': 1, '3': 0, '4': 1, '5': 1})
            self.assertEqual(result['pooled']['with_uncertainty_notes'], 1)
            for call, row in zip(client.calls, read_jsonl(run / 'inputs.jsonl')):
                self.assertNotIn('temperature', call)
                self.assertEqual(
                    call['messages'][0], {'role': 'system', 'content': DEFAULT_PROMPT.read_bytes().decode()}
                )
                self.assertEqual(len(call['messages']), 2)
                pair = json.loads(call['messages'][1]['content'])
                self.assertEqual(set(pair), {'source_text', 'asr_transcript'})
                self.assertEqual(pair['source_text'], row['text'])
                self.assertEqual(pair['asr_transcript'], read_json(run / 'asr' / f"{row['key']}.json")['transcript'])
                self.assertNotIn('SECRET QUESTION', call['messages'][1]['content'])
            self.assertGreater(len(json.loads(client.calls[1]['messages'][1]['content'])['source_text']), 8000)
            judge_semantic(run, client=client)
            self.assertEqual(len(client.calls), 3)
            changed = root / 'different_prompt.txt'
            changed.write_text('A different prompt')
            with self.assertRaisesRegex(ValueError, 'changed'):
                judge_semantic(run, prompt_path=changed, client=client)
            self.assertEqual(len(client.calls), 3)

    def test_retry_does_not_rewrite_the_pair_or_accept_markdown(self):
        def respond(kwargs, count):
            source = json.loads(kwargs['messages'][1]['content'])['source_text']
            if source.startswith('She'):
                result = grade(
                    4,
                    {
                        'type': 'minor_nuance_changed',
                        'source_evidence': 'carefully',
                        'asr_evidence': None,
                        'explanation': 'Manner omitted.',
                    },
                )
            elif source.startswith('The mass'):
                result = grade(
                    2,
                    {
                        'type': 'unit_or_currency_changed',
                        'source_evidence': '8 mg',
                        'asr_evidence': 'eight grams',
                        'explanation': 'Mass changed.',
                    },
                )
            else:
                result = grade(5)
            return '```json\n' + json.dumps(result) + '\n```' if count == 1 else result

        with tempfile.TemporaryDirectory() as tmp, patch('speech_roundtrip.semantic.time.sleep'):
            run = self.make_run(Path(tmp))
            client = FakeClient(respond)
            judge_semantic(run, client=client)
            self.assertEqual(len(client.calls), 4)
            self.assertEqual(client.calls[0], client.calls[1])
            first = next(read_jsonl(run / 'inputs.jsonl'))
            entry = read_json(run / 'semantic_judge/cache' / first['dataset'] / f"{first['key']}.json")
            self.assertEqual(len(entry['attempt_history']), 2)
            self.assertIn('JSONDecodeError', entry['attempt_history'][0]['error'])

    def test_invalid_output_is_not_converted_to_a_placeholder_score(self):
        with tempfile.TemporaryDirectory() as tmp, patch('speech_roundtrip.semantic.time.sleep'):
            run = self.make_run(Path(tmp))
            client = FakeClient(lambda *_: grade(None))
            with self.assertRaisesRegex(ValueError, 'Missing semantic score'):
                judge_semantic(run, client=client)
            status = read_json(run / 'semantic_judge/status.json')
            self.assertEqual(status, {'expected': 3, 'complete': 0, 'failed': 3})
            self.assertFalse((run / 'semantic_judge/evaluation_complete.txt').exists())
            for p in (run / 'semantic_judge/cache').rglob('*.json'):
                entry = read_json(p)
                self.assertEqual(entry['status'], 'failed')
                self.assertEqual(len(entry['attempt_history']), 4)


if __name__ == '__main__':
    unittest.main()
