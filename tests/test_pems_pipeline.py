"""Meaningful isolation, failure and compatibility checks for the new runner."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_pems_bay_pipeline as pipeline


class PipelineSafetyTests(unittest.TestCase):
    def test_original_outputs_and_parent_roots_are_rejected(self):
        for value in (ROOT, ROOT.parent, ROOT / 'outputs/pems-bay-evaluation', ROOT / 'runs'):
            with self.assertRaises(ValueError):
                pipeline.safe_destination(value)
        self.assertEqual(pipeline.safe_destination(ROOT / 'runs/new-run'), (ROOT / 'runs/new-run').resolve())

    def test_manifest_paths_cannot_escape(self):
        for name in ('../outside.py', '..', '.'):
            with self.assertRaises(ValueError):
                pipeline.checked_path(ROOT, name)

    def test_resume_cannot_skip_a_failed_stage_or_claim_fresh_confirmation(self):
        correct = {'completed_steps': ['design', 'binding'],
                   'dataset_already_analyzed': True, 'fresh_independent_validation': False}
        pipeline.validate_state(correct)
        for changed in ({**correct, 'completed_steps': ['binding']},
                        {**correct, 'completed_steps': ['design', 'reader_audit']},
                        {**correct, 'fresh_independent_validation': True}):
            with self.assertRaises(ValueError):
                pipeline.validate_state(changed)

    def test_changed_source_fails_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'source.py'
            path.write_text('original', encoding='utf-8')
            entries = {'source.py': {'bytes': path.stat().st_size, 'sha256': pipeline.sha(path)}}
            pipeline.verify_files(root, entries)
            path.write_text('modified', encoding='utf-8')
            with self.assertRaises(ValueError):
                pipeline.verify_files(root, entries)

    def test_existing_workspace_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preserved = root / 'important.txt'
            preserved.write_text('keep', encoding='utf-8')
            with self.assertRaises(ValueError):
                pipeline.initialize(root, root / 'missing.h5')
            self.assertEqual(preserved.read_text(encoding='utf-8'), 'keep')

    def test_stage_error_is_logged_and_stops_before_later_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'pipeline_records/logs').mkdir(parents=True)
            state = {'completed_steps': [], 'step_runs': [], 'status': 'ready'}
            process = types.SimpleNamespace(wait=lambda: 7)
            with patch.object(pipeline, 'STEPS', (('first', ('first.py',)), ('later', ('later.py',)))), \
                    patch.object(pipeline, 'stage_fingerprint'), \
                    patch.object(pipeline.subprocess, 'Popen', return_value=process) as launch:
                with self.assertRaises(RuntimeError):
                    pipeline.run_steps(root, state)
            self.assertEqual(launch.call_count, 1)
            saved = json.loads((root / 'pipeline_records/state.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['status'], 'failed')
            self.assertEqual(saved['completed_steps'], [])
            self.assertEqual(saved['step_runs'][0]['exit_code'], 7)
            self.assertFalse((root / 'pipeline_records/running.lock').exists())

    def test_lock_refuses_a_second_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'pipeline_records').mkdir()
            (root / 'pipeline_records/running.lock').write_text('existing owner', encoding='utf-8')
            with self.assertRaises(ValueError):
                pipeline.run_steps(root, {'completed_steps': [], 'step_runs': []})
            self.assertEqual((root / 'pipeline_records/running.lock').read_text(), 'existing owner')

    def test_float_lists_allow_rounding_but_not_changed_counts_or_structure(self):
        original = {'count': 3, 'ci': [0.1, 0.2]}
        self.assertEqual(pipeline.numeric_difference(original, {'count': 3, 'ci': [0.1 + 1e-14, 0.2]}), [])
        for changed in ({'count': 4, 'ci': [0.1, 0.2]}, {'count': 3, 'ci': [0.1]},
                        {'count': 3, 'ci': [0.11, 0.2]}, {'ci': [0.1, 0.2]}):
            self.assertTrue(pipeline.numeric_difference(original, changed))

    def test_risk_adapter_does_not_change_nonempty_scores_or_weights(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'outputs/pems-bay-risk'
            output.mkdir(parents=True)
            (root / 'complete_pems_bay_risk.py').write_text('compatibility marker', encoding='utf-8')
            for index in range(11):
                (output / f'weight{index}.pt').write_bytes(bytes([index]))
            original_hashes = {p.name: pipeline.sha(p) for p in output.glob('*.pt')}
            calls = []
            def original(model, x):
                calls.append(len(x))
                if not len(x):
                    raise AssertionError('Legacy empty concatenation must not execute')
                return x.sum(axis=1).astype(np.float32)
            fake = types.SimpleNamespace(OUTPUT=output, score=original)
            def main():
                empty = fake.score(None, np.empty((0, 51), dtype=np.float32))
                self.assertEqual(empty.shape, (0,))
                self.assertEqual(empty.dtype, np.float32)
                nonempty = np.array([[2., 3.], [4., 5.]], dtype=np.float32)
                np.testing.assert_array_equal(fake.score(None, nonempty), np.array([5., 9.], dtype=np.float32))
            fake.main = main
            with patch.dict(sys.modules, {'train_pems_bay_risk': fake}), patch.object(pipeline, 'ROOT', root):
                pipeline.risk_worker()
            self.assertEqual(calls, [2])
            self.assertIs(fake.score, original)
            record = json.loads((output / 'empty_group_adapter.json').read_text(encoding='utf-8'))
            self.assertEqual(record['preserved_weight_sha256'], original_hashes)
            self.assertFalse(record['training_and_model_selection_changed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
