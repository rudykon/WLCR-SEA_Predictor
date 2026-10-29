from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from tools.build_review_package import PREFIX, collect_files, safe_path, write_archive


class ReviewPackageTest(unittest.TestCase):
    def test_private_paths_and_history_are_excluded(self) -> None:
        for name in ('.git/config', '.github/workflows/demo.yml', 'data/train_data.csv',
                     'artifacts/report.csv', 'config/private/key.json', '.env',
                     '../outside', '/absolute', 'SOURCE_REVISION'):
            self.assertFalse(safe_path(name), name)
        self.assertTrue(safe_path('demo/checkpoints/A6_mixed_aug_seed42.pt'))
        self.assertTrue(safe_path('docs/assets/local-demo/weights.bin'))

    def test_archive_has_reproducible_bytes_and_content_checksums(self) -> None:
        files = {'CITATION.cff': b'authors:\n  - name: Anonymous Contributors\n',
                 'LICENSE': b'License text', 'sample.txt': b'unchanged\r\nbytes\r\n'}
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / 'first.zip'
            second = Path(directory) / 'second.zip'
            write_archive(files, first)
            write_archive(files, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with ZipFile(first) as archive:
                manifest = json.loads(archive.read(PREFIX + 'REVIEW_MANIFEST.json'))
                self.assertEqual(len(manifest['files']), len(files))
                self.assertEqual(archive.read(PREFIX + 'sample.txt'), files['sample.txt'])
                self.assertFalse(archive.comment)
                self.assertTrue(all(info.date_time == (2026,1,1,0,0,0) for info in archive.infolist()))
                self.assertTrue(all(not info.comment and not info.extra for info in archive.infolist()))

    def test_identity_markers_are_rejected_without_including_a_denylist(self) -> None:
        files = {'CITATION.cff': b'name: Anonymous Contributors', 'notes.txt': b'private-marker'}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'identity marker'):
                write_archive(files, Path(directory)/'review.zip', ('private-marker',))

    def test_source_collection_cannot_include_untracked_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'.git').mkdir()
            (root/'data').mkdir()
            (root/'data/train.csv').write_text('private local dataset')
            with patch('tools.build_review_package.subprocess.check_output', return_value=b'data/train.csv\0'):
                with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                    collect_files(root, include_site=False)


if __name__ == '__main__':
    unittest.main()
