# SPDX-FileCopyrightText: Copyright (c) 2026, Alexander Salikov.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for artifact publication and dataset integrity; no cloud requests."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf

from train import load_config, prepare_an4, publish_artifacts, write_transcription


class ArtifactTests(unittest.TestCase):
    """Exercise the success marker and failure behavior without importing NeMo."""

    def test_dataset_config_survives_detachment_for_reloaded_model(self) -> None:
        cfg = load_config(Path('/tmp/train.json'), Path('/tmp/test.json'))
        detached = OmegaConf.create(OmegaConf.to_container(cfg.model.test_ds, resolve=False))
        self.assertEqual(detached.labels, cfg.model.labels)
        self.assertEqual(detached.sample_rate, 16000)
        self.assertEqual(detached.manifest_filepath, '/tmp/test.json')

    def test_hypothesis_is_serialized_as_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'transcription.json'
            for text in ['recognized speech', '']:
                write_transcription('reference', SimpleNamespace(text=text), destination)
                self.assertEqual(json.loads(destination.read_text()), {'reference': 'reference', 'prediction': text})

    def test_published_hash_matches_downloaded_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source'
            source.mkdir()
            (source / 'model.nemo').write_bytes(b'model fixture')
            destination = Path(directory) / 'outputs'
            publish_artifacts(source, destination)
            hashes = json.loads((destination / 'COMPLETE.json').read_text())
            self.assertEqual(
                hashes['model.nemo'], hashlib.sha256((destination / 'model.nemo').read_bytes()).hexdigest()
            )

    def test_existing_artifacts_are_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            (destination / 'COMPLETE.json').write_text('original')
            with self.assertRaises(FileExistsError):
                publish_artifacts(destination, destination)
            self.assertEqual((destination / 'COMPLETE.json').read_text(), 'original')

    def test_failed_copy_does_not_publish_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source'
            source.mkdir()
            (source / 'model.nemo').write_bytes(b'model fixture')
            destination = Path(directory) / 'outputs'
            with patch('train.shutil.copyfile', side_effect=OSError('storage unavailable')):
                with self.assertRaises(OSError):
                    publish_artifacts(source, destination)
            self.assertFalse((destination / 'COMPLETE.json').exists())

    def test_bad_dataset_checksum_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'an4.tar.gz'
            archive.write_bytes(b'not AN4')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                prepare_an4(archive, Path(directory))


if __name__ == '__main__':
    unittest.main()
