"""Native NeMo Parakeet TDT ASR, retaining raw and English-normalized text."""

import hashlib
import importlib.metadata
import inspect
import json
import time
from pathlib import Path

from .asr import _audio_parts
from .core import atomic_bytes, digest, file_sha, read_json, read_jsonl, wav_info, write_json


def _native_runtime(spec):
    import torch
    from omegaconf import OmegaConf

    from nemo.collections.asr.models import ASRModel
    from nemo.collections.asr.parts.mixins.transcription import TranscribeConfig
    from nemo.collections.tts.parts.utils.tts_dataset_utils import get_text_processor

    if not torch.cuda.is_available():
        raise RuntimeError('Parakeet inference requires the allocated CUDA GPU')
    checkpoint = Path(spec['checkpoint'])
    before = checkpoint.stat()
    checksum = file_sha(checkpoint)
    print('PARAKEET_RESTORE', str(checkpoint), flush=True)
    model = ASRModel.restore_from(restore_path=str(checkpoint)).to('cuda').eval()
    processor = get_text_processor('en')
    if getattr(processor, 'normalizer', None) is None:
        raise RuntimeError('The requested English normalization requires nemo_text_processing and pynini')
    after = checkpoint.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Checkpoint changed during model restoration')
    metadata = {
        'checkpoint_sha256': checksum,
        'checkpoint_bytes': before.st_size,
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(),
        'model_class': type(model).__name__,
        'model_dtype': str(next(model.parameters()).dtype),
        'sample_rate': model.cfg.preprocessor.sample_rate,
        'decoding': OmegaConf.to_container(model.cfg.decoding, resolve=True),
        'processor': type(processor).__module__ + '.' + type(processor).__name__,
        'processor_source_sha256': file_sha(inspect.getfile(type(processor))),
        'text_normalization_version': importlib.metadata.version('nemo_text_processing'),
        'pynini_version': importlib.metadata.version('pynini'),
        'implementation_sha256': file_sha(__file__),
        'transcribe_config': {'batch_size': spec['batch_size'], 'use_lhotse': False},
    }
    try:
        metadata['nemo_version'] = importlib.metadata.version('nemo_toolkit')
    except importlib.metadata.PackageNotFoundError:
        metadata['nemo_version'] = None
    print('PARAKEET_RUNTIME', json.dumps(metadata), flush=True)

    def decode(paths):
        # The restore happens once; each batch gets fresh transcription bookkeeping.
        config = TranscribeConfig(batch_size=spec['batch_size'], use_lhotse=False)
        with torch.inference_mode():
            return model.transcribe(paths, override_config=config)

    return decode, processor.process_text_for_wer, metadata


def _cached_part(path, fingerprint):
    if not path.exists():
        return None
    saved = read_json(path)
    if saved['fingerprint'] != fingerprint:
        raise ValueError('Parakeet segment cache identity mismatch')
    if saved['status'] != 'complete':
        return None
    if hashlib.sha256(saved['raw_asr_text'].encode()).hexdigest() != saved['raw_asr_text_sha256']:
        raise ValueError('Cached Parakeet segment text was changed')
    return saved


def transcribe_parakeet(run_dir, limit=0, *, runtime=None):
    run = Path(run_dir).resolve()
    manifest = read_json(run / 'run_manifest.json')
    spec = manifest['config']['asr']
    if manifest.get('artifact_kind') != 'asr_variant' or spec.get('backend') != 'parakeet_tdt':
        raise ValueError('Parakeet requires a prepared, isolated ASR variant')
    if limit < 0 or file_sha(run / 'inputs.jsonl') != manifest['variant_identity']['input_sha256']:
        raise ValueError('Invalid selection or changed ASR inputs')
    rows = list(read_jsonl(run / 'inputs.jsonl'))
    if len(rows) != manifest['records']:
        raise ValueError('Parakeet source denominator changed')
    if limit:
        rows = rows[:limit]
    source = Path(manifest['variant_identity']['source_run'])
    if file_sha(source / 'run_manifest.json') != manifest['variant_identity']['source_manifest_sha256']:
        raise ValueError('The source run manifest changed')
    checkpoint_stat = Path(spec['checkpoint']).stat()
    if {'bytes': checkpoint_stat.st_size, 'mtime_ns': checkpoint_stat.st_mtime_ns} != manifest['variant_identity'][
        'checkpoint_stat'
    ]:
        raise ValueError('Parakeet checkpoint was replaced')
    decode, normalize, metadata = runtime if runtime is not None else _native_runtime(spec)
    receipt = {'identity': metadata, 'fingerprint': digest(metadata)}
    receipt_path = run / 'parakeet_runtime.json'
    if receipt_path.exists() and read_json(receipt_path) != receipt:
        raise ValueError('Parakeet runtime changed; use a new variant directory')
    write_json(receipt_path, receipt)
    plans, pending = [], []
    tts_hashes = dict(manifest['variant_identity']['tts_sidecars'])
    for row in rows:
        if file_sha(run / 'tts' / (row['key'] + '.json')) != tts_hashes[row['key'] + '.json']:
            raise ValueError('Frozen TTS metadata changed')
        tts = read_json(run / 'tts' / (row['key'] + '.json'))
        if tts['status'] != 'complete' or tts['text_sha256'] != hashlib.sha256(row['text'].encode()).hexdigest():
            raise ValueError('Parakeet source text no longer matches finalized TTS')
        audio = Path(tts['audio_path'])
        info = wav_info(audio, include_rms=False)
        if info['sha256'] != tts['audio']['sha256']:
            raise ValueError('Finalized TTS audio changed before Parakeet inference')
        parts = _audio_parts(source, row, tts, audio, info, spec)
        identity = digest([manifest['fingerprint'], receipt['fingerprint'], row['key'], info['sha256'], parts])
        plan = {'row': row, 'info': info, 'parts': parts, 'fingerprint': identity}
        for index, part in enumerate(parts):
            part['index'] = index
            part['fingerprint'] = digest([identity, index, part['audio_sha256']])
            part['cache_path'] = str(run / 'asr_chunks' / row['key'] / f'{index:04d}.json')
            if not _cached_part(Path(part['cache_path']), part['fingerprint']):
                pending.append(part)
        plans.append(plan)
    # Group similar lengths to avoid padding every ordinary clip to the longest reply.
    pending.sort(key=lambda p: p['frames'], reverse=True)
    print(
        'PARAKEET_PLAN',
        len(plans),
        'replies',
        sum(len(p['parts']) for p in plans),
        'segments',
        len(pending),
        'pending',
        flush=True,
    )
    for offset in range(0, len(pending), spec['batch_size']):
        batch = pending[offset : offset + spec['batch_size']]
        for part in batch:
            if file_sha(part['audio_path']) != part['audio_sha256']:
                raise ValueError('Parakeet input audio changed immediately before decoding')
        started = time.monotonic()
        hypotheses = decode([p['audio_path'] for p in batch])
        if not isinstance(hypotheses, (list, tuple)) or len(hypotheses) != len(batch):
            raise ValueError('Parakeet output count differs from input audio count')
        texts = [getattr(h, 'text', h) for h in hypotheses]
        if any(not isinstance(text, str) for text in texts):
            raise ValueError('Parakeet returned a non-text hypothesis')
        for part, text in zip(batch, texts):
            write_json(
                part['cache_path'],
                {
                    'fingerprint': part['fingerprint'],
                    'status': 'complete',
                    'audio_filepath': part['audio_path'],
                    'audio_sha256': part['audio_sha256'],
                    'raw_asr_text': text,
                    'raw_asr_text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                    'frames': part['frames'],
                    'batch_elapsed_s': time.monotonic() - started,
                },
            )
        print('PARAKEET_BATCH', offset + len(batch), '/', len(pending), 'new segments', flush=True)
    transcripts = []
    for plan in plans:
        row, info = plan['row'], plan['info']
        segments = [_cached_part(Path(p['cache_path']), p['fingerprint']) for p in plan['parts']]
        if any(s is None for s in segments) or sum(s['frames'] for s in segments) != info['frames']:
            raise ValueError('Parakeet did not cover all finalized speech frames')
        raw = ' '.join(s['raw_asr_text'].strip() for s in segments).strip()
        pred = normalize(raw)
        if not isinstance(pred, str):
            raise ValueError('English postprocessor did not return text')
        value = {
            'id': row['id'],
            'key': row['key'],
            'dataset': row['dataset'],
            'source_status': row['source_status'],
            'status': 'complete',
            'fingerprint': plan['fingerprint'],
            'model': manifest['evaluation_name'],
            'checkpoint': spec['checkpoint'],
            'checkpoint_sha256': metadata['checkpoint_sha256'],
            'audio_filepath': read_json(run / 'tts' / (row['key'] + '.json'))['audio_path'],
            'audio_sha256': info['sha256'],
            'raw_asr_text': raw,
            'pred_text': pred,
            'transcript': pred,
            'raw_asr_text_sha256': hashlib.sha256(raw.encode()).hexdigest(),
            'pred_text_sha256': hashlib.sha256(pred.encode()).hexdigest(),
            'transcript_sha256': hashlib.sha256(pred.encode()).hexdigest(),
            'transcript_field': 'pred_text',
            'normalization_scope': 'whole_reply_after_joining_segment_text',
            'words': len(pred.split()),
            'segmentation': 'whole_reply' if len(segments) == 1 else 'lossless_tts_chunks',
            'frames': info['frames'],
            'segments': segments,
        }
        target = run / 'asr' / (row['key'] + '.json')
        if target.exists() and read_json(target) != value:
            raise ValueError('A completed Parakeet reply was changed')
        if not target.exists():
            write_json(target, value)
        transcripts.append(value)
    if not limit:
        atomic_bytes(
            run / 'transcriptions.jsonl',
            ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in transcripts).encode(),
        )
        write_json(
            run / 'asr_status.json',
            {
                'name': manifest['evaluation_name'],
                'expected': manifest['records'],
                'complete': len(transcripts),
                'segments': sum(len(p['parts']) for p in plans),
                'raw_and_normalized_saved': True,
                'every_audio_frame_covered': True,
            },
        )
        atomic_bytes(
            run / 'asr_complete.txt', b'All selected speech transcribed; raw and English-normalized text retained.\n'
        )
    return len(transcripts)
