# SPDX-FileCopyrightText: Copyright (c) 2026, Alexander Salikov.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for artifact publication and dataset integrity; no cloud requests."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from train import prepare_an4, publish_artifacts


class ArtifactTests(unittest.TestCase):
    """Exercise the success marker and failure behavior without importing NeMo."""

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
