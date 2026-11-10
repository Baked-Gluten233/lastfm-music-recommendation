"""Regression tests for own-user exclusion and isolated validation experiments."""
from pathlib import Path
from unittest import mock
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

try:
    import torch
    from musicrec.content_feature_audit import (
        SOURCE_FILES, assign_user_folds, build_feature_banks, load_inputs, policy_data,
        replay_checkpoints, run_audit,
    )
    from musicrec.content import build_tag_features
    from musicrec.data import save_json, sha256
    from musicrec.neural import ContentTwoTower, CrossFittedContentTwoTower, score_catalogue
    from musicrec.phase2 import make_model, train_config
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Install optional PyTorch CPU dependency')
class CrossFitTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.train = pd.DataFrame({
            'userID': np.repeat(np.arange(1, 7), 2),
            'artistID': np.tile([10, 11], 6), 'weight': np.ones(12),
        })
        self.events = self.train[['userID', 'artistID']].copy()
        self.events['tagID'] = np.repeat(np.arange(20, 26), 2)
        self.artists = np.array([10, 11, 12, 13])
        self.namespace = np.arange(20, 30)

    def banks(self, events=None, train=None):
        return build_feature_banks(
            self.events if events is None else events,
            self.train if train is None else train,
            self.artists, self.namespace, folds=3, min_df=1, max_features=10, max_tags=10)

    def test_fold_assignment_is_balanced_and_input_order_independent(self):
        users = np.array([9, 3, 7, 2, 11, 1, 5])
        folds = assign_user_folds(users, folds=3)
        reversed_folds = assign_user_folds(users[::-1], folds=3)
        np.testing.assert_array_equal(folds, reversed_folds[::-1])
        self.assertLessEqual(np.ptp(np.bincount(folds)), 1)
        with self.assertRaises(ValueError):
            assign_user_folds([1, 1], folds=2)
        with self.assertRaises(ValueError):
            assign_user_folds([1, 2], folds=3)

    def test_own_fold_add_remove_and_duplicate_events_have_no_effect(self):
        before, audits, vocabularies = self.banks()
        fold = int(before['user_folds'][0])
        excluded_users = before['user_ids'][before['user_folds'] == fold]
        additions = pd.DataFrame({'userID': [1, 1], 'artistID': [10, 11], 'tagID': [29, 29]})
        poisoned = pd.concat([self.events, additions, additions], ignore_index=True)
        removed = self.events.loc[~self.events.userID.isin(excluded_users)]
        for events in (poisoned, removed):
            after, after_audits, after_vocabularies = self.banks(events)
            np.testing.assert_array_equal(before['crossfit_ids'][fold], after['crossfit_ids'][fold])
            np.testing.assert_array_equal(before['crossfit_weights'][fold], after['crossfit_weights'][fold])
            np.testing.assert_array_equal(before['namespace'], after['namespace'])
            self.assertEqual(audits['crossfit'][fold], after_audits['crossfit'][fold])
            self.assertEqual(vocabularies['crossfit'][fold], after_vocabularies['crossfit'][fold])
        changed, _, _ = self.banks(poisoned)
        self.assertFalse(np.array_equal(before['global_ids'], changed['global_ids']))

    def test_donor_tags_can_change_receiving_fold(self):
        before, _, _ = self.banks()
        fold = int(before['user_folds'][0])
        donor = int(before['user_ids'][before['user_folds'] != fold][0])
        event = pd.DataFrame({'userID': [donor], 'artistID': [10], 'tagID': [29]})
        after, _, _ = self.banks(pd.concat([self.events, event], ignore_index=True))
        self.assertFalse(np.array_equal(before['crossfit_ids'][fold], after['crossfit_ids'][fold]))

    def test_heldout_outside_catalogue_and_duplicate_tags(self):
        before, _, _ = self.banks()
        poison = pd.DataFrame({'userID': [1, 1], 'artistID': [12, 999], 'tagID': [99, 99]})
        events = pd.concat([self.events, self.events, poison], ignore_index=True)
        after, _, _ = self.banks(events)
        for key in ('global_ids', 'global_weights', 'crossfit_ids', 'crossfit_weights'):
            np.testing.assert_array_equal(before[key], after[key])

    def test_local_selection_precedes_static_coordinate_mapping(self):
        banks, _, _ = self.banks()
        ids, weights, vocabulary, _ = build_tag_features(
            self.events, self.train, self.artists, min_df=1, max_features=10, max_tags=10)
        mapping = np.append(np.searchsorted(self.namespace, vocabulary), len(self.namespace))
        np.testing.assert_array_equal(banks['global_ids'][0], mapping[ids])
        np.testing.assert_array_equal(banks['global_weights'][0], weights)
        np.testing.assert_array_equal(banks['global_ids'][0, 2], np.full(10, len(self.namespace)))
        np.testing.assert_array_equal(banks['global_weights'][0, 2], [1.] + [0.] * 9)
        with self.assertRaises(ValueError):
            build_feature_banks(self.events, self.train, self.artists, self.namespace[::-1], folds=3)
        with self.assertRaises(ValueError):
            build_feature_banks(self.events, self.train, self.artists, np.array([29]), folds=3, min_df=1)

    def test_policy_models_have_identical_initial_parameters_and_correct_scores(self):
        banks, _, _ = self.banks()
        models = []
        for policy in ('global', 'crossfit'):
            current = policy_data({}, banks, policy)
            torch.manual_seed(42)
            model = CrossFittedContentTwoTower(
                6, current['feature_ids'], current['feature_weights'], len(self.namespace),
                current['user_folds'], dim=4)
            models.append(model)
            users = torch.tensor([0, 1, 2, 3, 4, 5])
            items = torch.tensor([0, 1, 2, 3, 0, 1])
            torch.testing.assert_close(model(users, items), model.all_scores()[users, items])
            actual = score_catalogue(model, 6, 4, np.array([1, 1, 0, 0], dtype=bool))
            self.assertTrue(np.isfinite(actual).all())
            self.assertTrue((actual[:, 2:] < actual[:, :2].min(axis=1, keepdims=True)).all())
            (-model(users, items)).mean().backward()
            self.assertGreater(float(model.tag_embedding.weight.grad.abs().sum()), 0)
        for left, right in zip(models[0].parameters(), models[1].parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_legacy_model_state_dict_layout_is_unchanged(self):
        ids, weights, vocabulary, _ = build_tag_features(
            self.events, self.train, self.artists, min_df=1)
        legacy = ContentTwoTower(6, ids, weights, len(vocabulary), dim=4)
        data = {'matrix': np.zeros((6, 4)), 'feature_ids': ids, 'feature_weights': weights,
                'vocabulary': vocabulary}
        rebuilt = make_model({'family': 'content_two_tower', 'dim': 4}, data)
        rebuilt.load_state_dict(legacy.state_dict(), strict=True)
        self.assertNotIn('user_folds', legacy.state_dict())
        torch.testing.assert_close(legacy.all_scores(), rebuilt.all_scores())

    def write_fixture(self, root):
        raw = root/'data/raw/hetrec2011-lastfm-2k'
        raw.mkdir(parents=True)
        train = pd.DataFrame({'userID': np.repeat(np.arange(1, 7), 3),
                              'artistID': np.tile([10, 11, 12, 10, 11, 13], 3), 'weight': np.ones(18)})
        validation = pd.DataFrame({'userID': np.arange(1, 7), 'artistID': [14]*6, 'weight': [1]*6})
        (root/'data/processed').mkdir(parents=True)
        train.to_csv(root/'data/processed/train.csv', index=False)
        validation.to_csv(root/'data/processed/validation.csv', index=False)
        pd.DataFrame({'id': np.arange(10, 34)}).to_csv(raw/'artists.dat', sep='\t', index=False)
        pd.DataFrame({'tagID': np.arange(20, 30)}).to_csv(raw/'tags.dat', sep='\t', index=False)
        events = train[['userID', 'artistID']].copy()
        events['tagID'] = np.repeat(np.arange(20, 26), 3)
        events.to_csv(raw/'user_taggedartists-timestamps.dat', sep='\t', index=False)
        save_json(root/'outputs/split_manifest.json', {'split_sha256': {
            name: sha256(root/f'data/processed/{name}.csv') for name in ('train', 'validation')}})
        save_json(root/'outputs/data_audit.json', {'raw_files_sha256': {
            p.name: sha256(p) for p in raw.iterdir()}})

    def test_default_cli_audit_needs_no_test_file_and_does_not_train(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('TEMP')) as directory:
            root = Path(directory)
            self.write_fixture(root)
            with mock.patch('musicrec.content_feature_audit.train_config') as training:
                audit = run_audit(root)
            training.assert_not_called()
            self.assertEqual(audit['training_pairs'], 18)
            self.assertFalse((root/'data/processed/test.csv').exists())
            status = json.loads((root/'outputs/content_feature_audit/status.json').read_text(encoding='utf-8'))
            self.assertEqual(status['training_runs'], 0)
            self.assertFalse(status['test_labels_read'])
            with self.assertRaises(ValueError):
                run_audit(root, root/'outputs/phase2')

    def test_crossfit_training_reproducibility_and_checkpoint_loading(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('TEMP')) as directory:
            root = Path(directory)
            self.write_fixture(root)
            data, events, namespace, _ = load_inputs(root)
            banks, _, _ = build_feature_banks(events, data['train'], data['artist_ids'], namespace)
            current = policy_data(data, banks, 'crossfit')
            config = {'family': 'content_two_tower_audit', 'dim': 4, 'lr': .003,
                      'feature_policy': 'crossfit'}
            destination = root/'runs'
            destination.mkdir()
            result = train_config(config, current, destination, seed=42, max_epochs=2, every=1)
            checkpoint = destination/(result['run'] + '_weights.npz')
            with np.load(checkpoint, allow_pickle=False) as saved:
                first = {name: saved[name].copy() for name in saved.files}
            repeated = train_config(config, current, destination, seed=42, max_epochs=2, every=1)
            with np.load(checkpoint, allow_pickle=False) as saved:
                for name in saved.files:
                    np.testing.assert_array_equal(first[name], saved[name])
            self.assertEqual(result['ndcg@10'], repeated['ndcg@10'])
            model = make_model(config, current)
            model.load_state_dict({name: torch.from_numpy(value) for name, value in first.items()}, strict=True)
            self.assertTrue(torch.isfinite(model.all_scores()).all())

    def test_six_checkpoint_replay_and_source_hash_guard(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('TEMP')) as directory:
            root = Path(directory)
            self.write_fixture(root)
            for name in SOURCE_FILES:
                target = root/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT/name).read_bytes())
            result = run_audit(root, run_validation=True)
            self.assertEqual(len(result), 6)
            output = root/'outputs/content_feature_audit'
            self.assertEqual(replay_checkpoints(root, output)['checkpoint_count'], 6)
            users = pd.read_csv(output/'global/content_two_tower_audit_lr0.003_seed42_users.csv')
            with mock.patch('musicrec.content_feature_audit.evaluate_model',
                            return_value=({'ndcg@10': float('nan')}, users)):
                with self.assertRaisesRegex(ValueError, 'replay mismatch'):
                    replay_checkpoints(root, output)
            source = root/'src/musicrec/neural.py'
            source.write_bytes(source.read_bytes() + b'\n# Changed after the run\n')
            with self.assertRaisesRegex(ValueError, 'source changed'):
                run_audit(root, replay_only=True)
            replay = json.loads((output/'replay.json').read_text(encoding='utf-8'))
            self.assertEqual(replay['status'], 'failed')

    def test_failed_training_has_explicit_status(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('TEMP')) as directory:
            root = Path(directory)
            self.write_fixture(root)
            for name in SOURCE_FILES:
                target = root/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT/name).read_bytes())
            with mock.patch('musicrec.content_feature_audit.train_config', side_effect=RuntimeError('fixture failure')):
                with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                    run_audit(root, run_validation=True)
            status = json.loads((root/'outputs/content_feature_audit/status.json').read_text(encoding='utf-8'))
            self.assertEqual(status['stage'], 'validation_failed')
            self.assertEqual(status['training_runs'], 0)


if __name__ == '__main__':
    unittest.main()
