# SPDX-FileCopyrightText: Copyright (c) 2026, Alexander Salikov.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, single-GPU Conformer CTC training on AN4 with portable artifacts."""

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

AN4_URL = 'https://dldata-public.s3.us-east-2.amazonaws.com/an4_sphere.tar.gz'
AN4_SHA256 = 'a0525579493735d32b60fb6fa5975dd0eaacde087b0a4eeb7f13b7ef33077b12'


def prepare_an4(archive: Path, work: Path) -> tuple[Path, Path]:
    """Verify AN4, convert SPHERE to WAV, and create absolute-path NeMo manifests."""
    import soundfile as sf

    if hashlib.sha256(archive.read_bytes()).hexdigest() != AN4_SHA256:
        raise ValueError('AN4 checksum mismatch; refusing to extract the archive')
    with tarfile.open(archive) as source:
        source.extractall(work, filter='data')
    manifests = []
    for split, audio_dir in [('train', 'an4_clstk'), ('test', 'an4test_clstk')]:
        manifest = work / 'an4' / f'{split}_manifest.json'
        transcripts = work / 'an4' / 'etc' / f'an4_{split}.transcription'
        with manifest.open('w') as output:
            for line in transcripts.read_text().splitlines():
                transcript, file_id = line.rsplit('(', 1)
                file_id = file_id.rstrip(')')
                speaker = file_id.split('-')[1]
                source = work / 'an4' / 'wav' / audio_dir / speaker / f'{file_id}.sph'
                samples, sample_rate = sf.read(source)
                if sample_rate != 16000 or samples.ndim != 1 or len(samples) == 0:
                    raise ValueError(f'Expected nonempty mono 16 kHz audio: {source}')
                wav = source.with_suffix('.wav')
                sf.write(wav, samples, sample_rate, subtype='PCM_16')
                text = transcript.replace('<s>', '').replace('</s>', '').strip().lower()
                record = dict(audio_filepath=str(wav.resolve()), duration=len(samples) / sample_rate, text=text)
                output.write(json.dumps(record) + '\n')
        manifests.append(manifest)
    return tuple(manifests)


def publish_artifacts(source: Path, destination: Path) -> None:
    """Copy closed local files, then write hashes last; failed copies have no success marker."""
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(f'Use a fresh output directory: {destination}')
    hashes = {}
    for path in sorted(source.iterdir()):
        if path.is_file():
            shutil.copyfile(path, destination / path.name)
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (destination / 'COMPLETE.json').write_text(json.dumps(hashes, indent=2) + '\n')


def load_config(train_manifest: Path, test_manifest: Path) -> DictConfig:
    """Fill manifest paths and resolve references before NeMo copies dataset sections."""
    cfg = OmegaConf.load(Path(__file__).with_name('conformer.yaml'))
    cfg.model.train_ds.manifest_filepath = str(train_manifest)
    cfg.model.validation_ds.manifest_filepath = str(test_manifest)
    cfg.model.test_ds.manifest_filepath = str(test_manifest)
    OmegaConf.resolve(cfg)
    return cfg


def write_transcription(reference: str, hypothesis: Any, destination: Path) -> None:
    """Persist the text of a NeMo Hypothesis, including a valid empty prediction."""
    result = dict(reference=reference, prediction=hypothesis.text)
    destination.write_text(json.dumps(result, indent=2) + '\n')


def train(work: Path, artifacts: Path, max_steps: int) -> None:
    """Train, reload the exported model, evaluate AN4 test data, and transcribe a sample."""
    import lightning.pytorch as pl
    import torch

    from nemo.collections.asr.models import EncDecCTCModel
    from nemo.utils import logging

    if not torch.cuda.is_available():
        raise RuntimeError('This training example requires one CUDA GPU')
    pl.seed_everything(42, workers=True)
    archive = work / 'an4_sphere.tar.gz'
    with urllib.request.urlopen(AN4_URL, timeout=60) as response, archive.open('wb') as output:
        shutil.copyfileobj(response, output)
    train_manifest, test_manifest = prepare_an4(archive, work)
    cfg = load_config(train_manifest, test_manifest)
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        num_nodes=1,
        precision='32-true',
        max_steps=max_steps,
        max_epochs=-1,
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        limit_val_batches=2,
        log_every_n_steps=10,
        default_root_dir=str(work),
    )
    model = EncDecCTCModel(cfg=cfg.model, trainer=trainer)
    trainer.fit(model)
    artifacts.mkdir()
    model.save_to(str(artifacts / 'model.nemo'))
    OmegaConf.save(cfg, artifacts / 'config.yaml')
    del model
    torch.cuda.empty_cache()
    restored = EncDecCTCModel.restore_from(str(artifacts / 'model.nemo'))
    restored.setup_test_data(cfg.model.test_ds)
    evaluator = pl.Trainer(accelerator='gpu', devices=1, logger=False, enable_checkpointing=False)
    metrics = evaluator.test(restored, verbose=False)
    if not metrics or 'test_wer' not in metrics[0]:
        raise RuntimeError('Evaluation did not return test_wer')
    (artifacts / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')
    record = json.loads(test_manifest.read_text().splitlines()[0])
    shutil.copyfile(record['audio_filepath'], artifacts / 'sample.wav')
    restored.eval()
    predictions = restored.transcribe([str(artifacts / 'sample.wav')], batch_size=1, return_hypotheses=True)
    write_transcription(record['text'], predictions[0], artifacts / 'transcription.json')
    (artifacts / 'run.json').write_text(json.dumps(dict(seed=42, steps=trainer.global_step, an4_sha256=AN4_SHA256)))
    logging.info('Reloaded model test metrics: %s', metrics)


def main() -> None:
    """Run a capped smoke training job; export only after successful reload and evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-steps', type=int, default=50)
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 1000:
        parser.error('--max-steps must be between 1 and 1000')
    output = os.environ.get('NEBIUS_OUTPUT_DIR')
    if not output:
        parser.error('NEBIUS_OUTPUT_DIR must point to the mounted output directory')
    destination = Path(output) / 'asr'
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f'Use a unique job name and fresh output prefix: {destination}')
    with tempfile.TemporaryDirectory(prefix='nemo-an4-') as directory:
        work = Path(directory)
        artifacts = work / 'artifacts'
        train(work, artifacts, args.max_steps)
        publish_artifacts(artifacts, destination)


if __name__ == '__main__':
    main()
