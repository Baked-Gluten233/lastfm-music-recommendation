"""Synthetic acceptance-runner regressions, without real test-set scoring."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import verify_windows as verification


class WindowsVerificationTests(unittest.TestCase):
    def test_explicit_temporary_directory_cannot_fall_back(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.dict(os.environ, {'TMPDIR': str(root / 'missing')}):
                with self.assertRaisesRegex(RuntimeError, 'temporary directory is unavailable'):
                    verification.validate_temp_directory()
            with patch.dict(os.environ, {'TMPDIR': str(root)}), \
                 patch.object(verification.tempfile, 'gettempdir', return_value=str(root / 'other')):
                with self.assertRaisesRegex(RuntimeError, 'fallback detected'):
                    verification.validate_temp_directory()
            with patch.dict(os.environ, {'TMPDIR': str(root)}), \
                 patch.object(verification.tempfile, 'TemporaryFile', side_effect=PermissionError('synthetic')):
                with self.assertRaisesRegex(RuntimeError, 'temporary directory is unavailable'):
                    verification.validate_temp_directory()

    def test_test_suite_records_counts_skips_and_optimization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'source').mkdir()
            output = root / 'results'
            output.mkdir()
            log = 'Ran 4 tests in 0.01s\n\nOK (skipped=1)\n'
            process = subprocess.CompletedProcess([], 0, '', log)
            workspaces = []

            def run(command, **kwargs):
                workspaces.append(kwargs['cwd'])
                self.assertIn('-O', command)
                return process

            with patch.object(verification.subprocess, 'run', side_effect=run):
                summary = verification.run_tests(root / 'source', output, os.environ.copy(), True)
                verification.run_tests(root / 'source', output, os.environ.copy(), True)
            self.assertEqual(summary['tests'], 4)
            self.assertEqual(summary['skipped'], 1)
            self.assertFalse(summary['passed'])
            self.assertIn('duration_seconds', summary)
            self.assertNotEqual(workspaces[0], workspaces[1])
            self.assertTrue(all(not path.exists() for path in workspaces))
            self.assertEqual((output / summary['log']).read_text(encoding='utf-8'), log)

    def test_empty_or_failed_test_suite_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for process in (subprocess.CompletedProcess([], 0, '', 'Ran 0 tests in 0s\nOK\n'),
                            subprocess.CompletedProcess([], 1, '', 'Ran 2 tests in 0s\nFAILED\n')):
                with patch.object(verification.subprocess, 'run', return_value=process):
                    self.assertFalse(verification.run_tests(root, root, {}, False)['passed'])

    def test_integrity_detects_added_removed_and_changed_files(self):
        result = verification.integrity_result({'a': 'old', 'b': 'same'}, {'a': 'new', 'c': 'new'})
        self.assertFalse(result['passed'])
        self.assertEqual(result['changed'], ['a'])
        self.assertEqual(result['removed'], ['b'])
        self.assertEqual(result['added'], ['c'])

    def test_audit_and_acceptance_results_do_not_change_protected_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = root / 'outputs/final'
            frozen.mkdir(parents=True)
            (frozen / 'saved.csv').write_text('frozen', encoding='utf-8')
            before = verification.file_hashes(root, ('data', 'notebooks', 'outputs'))
            for name in ('content_feature_audit', 'windows_native_verification'):
                output = root / 'outputs' / name
                output.mkdir()
                (output / 'new.json').write_text('{}', encoding='utf-8')
            self.assertEqual(before, verification.file_hashes(root, ('data', 'notebooks', 'outputs')))
            (frozen / 'saved.csv').write_text('changed', encoding='utf-8')
            self.assertNotEqual(before, verification.file_hashes(root, ('data', 'notebooks', 'outputs')))

    def test_source_digest_tracks_runtime_code_and_pins(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'src').mkdir()
            (root / 'src/code.py').write_text('value = 1\n', encoding='utf-8')
            for filename in ('requirements.txt', 'requirements-phase2.txt'):
                (root / filename).write_text('', encoding='utf-8')
            before = verification.source_manifest(root)
            (root / 'status.json').write_text('{}', encoding='utf-8')
            self.assertEqual(before, verification.source_manifest(root))
            (root / 'requirements.txt').write_text('numpy==1\n', encoding='utf-8')
            self.assertNotEqual(before, verification.source_manifest(root))

    def test_dependency_pin_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'requirements.txt').write_text('numpy==2.3.5\n', encoding='utf-8')
            (root / 'requirements-phase2.txt').write_text('torch==2.9.1\n', encoding='utf-8')
            verification.verify_dependency_pins(root, {'numpy': '2.3.5', 'torch': '2.9.1+cpu'})
            with self.assertRaisesRegex(RuntimeError, 'Pinned dependency mismatch'):
                verification.verify_dependency_pins(root, {'numpy': '2.0', 'torch': '2.9.1+cpu'})

    def test_notebook_guard_blocks_final_scoring_entrypoints(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
        import musicrec.final_evaluation as final
        original_entry = final.run_final_once
        original_scorer = final.evaluate
        try:
            exec(compile(verification.FINAL_TEST_GUARD, 'acceptance-guard', 'exec', optimize=2), {})
            with self.assertRaisesRegex(RuntimeError, 'Held-out test scoring is disabled'):
                final.run_final_once(Path('synthetic-only'))
            with self.assertRaisesRegex(RuntimeError, 'Held-out test scoring is disabled'):
                final.evaluate()
        finally:
            final.run_final_once = original_entry
            final.evaluate = original_scorer

    def test_failed_run_still_records_integrity_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for filename in ('requirements.txt', 'requirements-phase2.txt'):
                (root / filename).write_text('', encoding='utf-8')
            frozen = root / 'data/frozen.csv'
            frozen.parent.mkdir()
            frozen.write_text('original', encoding='utf-8')

            def failed_check(*args):
                frozen.write_text('changed', encoding='utf-8')
                raise RuntimeError('Synthetic dependency failure')

            with patch.object(verification, 'ROOT', root), \
                 patch.object(sys, 'argv', ['verify_windows']), \
                 patch.object(verification.platform, 'system', return_value='Windows'), \
                 patch.object(verification, 'run_command', side_effect=failed_check):
                with self.assertRaisesRegex(RuntimeError, 'Synthetic dependency failure'):
                    verification.main()
            status = json.loads((root / 'outputs/windows_native_verification/status.json').read_text(encoding='utf-8'))
            self.assertEqual(status['status'], 'failed')
            self.assertFalse(status['native_kernel_verified'])
            self.assertFalse(status['integrity']['passed'])
            self.assertEqual(status['integrity']['changed'], ['data/frozen.csv'])

    def test_successful_steps_cannot_hide_integrity_or_source_changes(self):
        for target in ('data/frozen.csv', 'requirements.txt'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                for filename in ('requirements.txt', 'requirements-phase2.txt', 'data/frozen.csv'):
                    path = root / filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('', encoding='utf-8')

                def execute(source, output, filename, env):
                    (root / target).write_text('changed', encoding='utf-8')
                    return {'name': filename, 'native_kernel': 'passed'}

                process = subprocess.CompletedProcess([], 0, '', '')
                suite = {'passed': True, 'tests': 1, 'skipped': 0, 'log': 'synthetic.log'}
                with patch.object(verification, 'ROOT', root), \
                     patch.object(sys, 'argv', ['verify_windows']), \
                     patch.object(verification.platform, 'system', return_value='Windows'), \
                     patch.object(verification, 'run_command', return_value=(process, {}, '')), \
                     patch.object(verification, 'run_tests', return_value=suite), \
                     patch.object(verification, 'execute_notebook', side_effect=execute):
                    with self.assertRaisesRegex(RuntimeError, 'changed during acceptance'):
                        verification.main()
                status = json.loads((root / 'outputs/windows_native_verification/status.json').read_text(encoding='utf-8'))
                self.assertEqual(status['status'], 'failed')
                self.assertIn('integrity_error', status)


if __name__ == '__main__':
    unittest.main()
