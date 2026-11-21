"""Exercise the actual sdist backend so shipped regression assets stay usable."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest


def _copy_source_tree(root, checkout):
    ignore_generated = shutil.ignore_patterns(
        '.git', '*.egg-info', '__pycache__', '.pytest_cache', 'dist', 'build'
    )

    def ignore(directory, names):
        skipped = ignore_generated(directory, names)
        # A developer's virtual environment is not source. Recognize its
        # marker rather than relying on a particular directory name.
        skipped.update(name for name in names
                       if (Path(directory) / name / 'pyvenv.cfg').is_file())
        return skipped

    shutil.copytree(root, checkout, ignore=ignore)


class SourceDistributionAssetsTests(unittest.TestCase):
    def test_staging_excludes_local_virtual_environments(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'source'
            root.mkdir()
            (root / 'keep.txt').write_text('source asset', encoding='utf-8')
            for name in ('.venv', 'custom-python-environment'):
                environment = root / name
                environment.mkdir()
                (environment / 'pyvenv.cfg').write_text('home = /example\n', encoding='utf-8')
                (environment / 'dependency.bin').write_bytes(b'not source')
            checkout = Path(folder) / 'checkout'
            _copy_source_tree(root, checkout)
            self.assertEqual((checkout / 'keep.txt').read_text(encoding='utf-8'), 'source asset')
            self.assertFalse((checkout / '.venv').exists())
            self.assertFalse((checkout / 'custom-python-environment').exists())

    def test_sdist_contains_the_tutorial_notebook_and_packaging_regression(self):
        root = Path(__file__).resolve().parents[1]
        notebook = 'tutorial/notebooks/01_run_models/01_single_scenario_partmc.ipynb'
        with tempfile.TemporaryDirectory() as folder:
            staging = Path(folder)
            checkout = staging / 'source'
            _copy_source_tree(root, checkout)
            output = staging / 'dist'
            output.mkdir()
            result = subprocess.run(
                [sys.executable, '-c',
                 'import sys; from setuptools.build_meta import build_sdist; build_sdist(sys.argv[1])',
                 str(output)], cwd=checkout, capture_output=True, text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            archives = list(output.glob('*.tar.gz'))
            self.assertEqual(len(archives), 1)
            with tarfile.open(archives[0]) as archive:
                members = {name.split('/', 1)[1]: name for name in archive.getnames()
                           if '/' in name}
                self.assertIn(notebook, members)
                self.assertIn('test/test_source_distribution.py', members)
                with archive.extractfile(members[notebook]) as stream:
                    contents = stream.read()
                self.assertEqual(contents, (root / notebook).read_bytes())
                self.assertIn('cells', json.loads(contents))
