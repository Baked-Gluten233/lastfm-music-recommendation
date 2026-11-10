"""Isolated train/validation audit of user-authored content features."""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .content import build_tag_features
from .data import indexed_matrix, save_json, sha256
from .models import popularity_scores
from .phase2 import evaluate_model, make_model, train_config


SEEDS = (42, 43, 44)
CONFIG = {'family': 'content_two_tower_audit', 'dim': 32, 'lr': .003}
SOURCE_FILES = (
    'src/musicrec/content_feature_audit.py', 'src/musicrec/content.py',
    'src/musicrec/neural.py', 'src/musicrec/phase2.py', 'src/musicrec/data.py',
    'src/musicrec/evaluation.py', 'src/musicrec/models.py',
    'src/musicrec/validation.py', 'scripts/audit_content_features.py',
)
INPUT_FILES = (
    'data/processed/train.csv', 'data/processed/validation.csv',
    'data/raw/hetrec2011-lastfm-2k/artists.dat',
    'data/raw/hetrec2011-lastfm-2k/tags.dat',
    'data/raw/hetrec2011-lastfm-2k/user_taggedartists-timestamps.dat',
    'outputs/split_manifest.json', 'outputs/data_audit.json',
)


def assign_user_folds(user_ids, folds=5, seed=42):
    """Balanced deterministic assignment, aligned to the supplied user order."""
    user_ids = np.asarray(user_ids, dtype=np.int64)
    if len(np.unique(user_ids)) != len(user_ids):
        raise ValueError('Fold assignment requires unique users')
    if folds < 2 or folds > len(user_ids):
        raise ValueError('Use at least two folds and at most one per user')
    order = np.argsort(user_ids)
    shuffled = np.random.default_rng(seed).permutation(len(user_ids))
    result = np.empty(len(user_ids), dtype=np.int64)
    result[order[shuffled]] = np.arange(len(user_ids)) % folds
    return result


def build_feature_banks(events, train, artist_ids, namespace, folds=5, seed=42,
                        min_df=2, max_features=2000, max_tags=32):
    """Fit each local vocabulary before mapping to the static tag coordinates.

    Excluded-user events cannot affect that bank's vocabulary, TF, DF, IDF,
    truncation or coordinate IDs. The fixed namespace comes only from tags.dat.
    """
    namespace = np.asarray(namespace, dtype=np.int64)
    if not np.array_equal(namespace, np.unique(namespace)):
        raise ValueError('The static tag namespace must be sorted and unique')
    user_ids = np.sort(train.userID.unique())
    user_folds = assign_user_folds(user_ids, folds, seed)
    fold_by_user = dict(zip(user_ids, user_folds))
    policies = {'global': [], 'crossfit': []}
    audits = {'global': [], 'crossfit': []}
    vocabularies = {'global': [], 'crossfit': []}
    for policy, excluded_folds in (('global', (None,)), ('crossfit', range(folds))):
        for fold in excluded_folds:
            donors = events
            if fold is not None:
                donors = events.loc[events.userID.map(fold_by_user) != fold]
            ids, weights, vocabulary, audit = build_tag_features(
                donors, train, artist_ids, min_df, max_features, max_tags)
            tag_ids = np.asarray(vocabulary, dtype=np.int64)
            if not np.isin(tag_ids, namespace).all():
                raise ValueError('An authorized feature tag is absent from tags.dat')
            # Remap after top-k selection to preserve legacy local tie ordering.
            coordinate_map = np.append(np.searchsorted(namespace, tag_ids), len(namespace))
            policies[policy].append((coordinate_map[ids].astype(np.int64), weights))
            excluded_events = 0
            if fold is not None:
                excluded_events = int((donors.userID.map(fold_by_user) == fold).sum())
            audits[policy].append({**audit, 'excluded_fold': fold,
                                   'excluded_user_events_in_donors': excluded_events})
            vocabularies[policy].append(vocabulary)
    banks = {
        'global_ids': np.stack([v[0] for v in policies['global']]),
        'global_weights': np.stack([v[1] for v in policies['global']]),
        'crossfit_ids': np.stack([v[0] for v in policies['crossfit']]),
        'crossfit_weights': np.stack([v[1] for v in policies['crossfit']]),
        'namespace': namespace, 'user_ids': user_ids, 'user_folds': user_folds,
        'artist_ids': np.asarray(artist_ids, dtype=np.int64),
    }
    return banks, audits, vocabularies


def audit_self_contribution(events, train, artist_ids, banks, vocabulary):
    """Count direct own-user support in the actual selected global features."""
    within = events.loc[events.artistID.isin(artist_ids), ['userID', 'artistID', 'tagID']]
    allowed = within.merge(train[['userID', 'artistID']], on=['userID', 'artistID'],
                           how='inner', validate='many_to_one').drop_duplicates()
    kept = allowed.loc[allowed.tagID.isin(vocabulary)]
    feature_ids = banks['global_ids'][0]
    feature_weights = banks['global_weights'][0]
    row_ids, positions = np.nonzero((feature_weights > 0) & (feature_ids < len(banks['namespace'])))
    selected = pd.DataFrame({
        'artistID': np.asarray(artist_ids)[row_ids],
        'tagID': banks['namespace'][feature_ids[row_ids, positions]],
    })
    providers = kept.groupby(['artistID', 'tagID']).size().rename('providers').reset_index()
    own = kept.merge(selected, on=['artistID', 'tagID'], validate='many_to_one')
    own = own.merge(providers, on=['artistID', 'tagID'], validate='many_to_one')
    selected_providers = selected.merge(providers, on=['artistID', 'tagID'])
    pairs_with_tags = len(allowed[['userID', 'artistID']].drop_duplicates())
    selected_pairs = len(own[['userID', 'artistID']].drop_duplicates())
    exclusive_pairs = len(own.loc[own.providers == 1, ['userID', 'artistID']].drop_duplicates())
    fold_by_user = dict(zip(banks['user_ids'], banks['user_folds']))
    fold_rows = []
    for fold in range(len(banks['crossfit_ids'])):
        receivers = train.loc[train.userID.map(fold_by_user) == fold]
        ids = banks['crossfit_ids'][fold]
        weights = banks['crossfit_weights'][fold]
        has_content = ((ids < len(banks['namespace'])) & (weights > 0)).any(axis=1)
        content_artists = np.asarray(artist_ids)[has_content]
        fold_rows.append({
            'fold': fold, 'users': int((banks['user_folds'] == fold).sum()),
            'training_pairs': len(receivers), 'artists_with_content': int(has_content.sum()),
            'training_pair_content_coverage': float(receivers.artistID.isin(content_artists).mean()),
        })
    return {
        'training_pairs': len(train), 'training_users': int(train.userID.nunique()),
        'authorized_unique_tag_events': len(allowed),
        'pairs_with_own_authorized_tags': pairs_with_tags,
        'own_tag_pair_fraction': pairs_with_tags / len(train),
        'pairs_with_own_tags_in_local_vocabulary': len(kept[['userID', 'artistID']].drop_duplicates()),
        'pairs_with_own_tags_in_top32': selected_pairs,
        'own_top32_pair_fraction': selected_pairs / len(train),
        'pairs_with_exclusive_own_tag_in_top32': exclusive_pairs,
        'exclusive_own_top32_pair_fraction': exclusive_pairs / len(train),
        'selected_artist_tag_features': len(selected),
        'selected_features_with_one_provider': int((selected_providers.providers == 1).sum()),
        'static_namespace_size': len(banks['namespace']), 'fold_coverage': fold_rows,
        'interpretation': 'Direct training-feature self-contribution exists; this does not establish test leakage or metric inflation.',
        'crossfit_limitation': 'Cross-fitting also reduces tag providers. Shared parameters can learn indirectly from other folds.',
    }


def load_inputs(root):
    """Read only the explicitly listed train/validation inputs and manifests."""
    root = Path(root)
    manifest = json.loads((root/'outputs/split_manifest.json').read_text(encoding='utf-8'))
    data_audit = json.loads((root/'outputs/data_audit.json').read_text(encoding='utf-8'))
    input_hashes = {name: sha256(root/name) for name in INPUT_FILES}
    for split in ('train', 'validation'):
        if input_hashes[f'data/processed/{split}.csv'] != manifest['split_sha256'][split]:
            raise ValueError('Frozen split has changed: ' + split)
    raw = root/'data/raw/hetrec2011-lastfm-2k'
    for name in ('artists.dat', 'tags.dat', 'user_taggedartists-timestamps.dat'):
        if input_hashes['data/raw/hetrec2011-lastfm-2k/' + name] != data_audit['raw_files_sha256'][name]:
            raise ValueError('Original input changed: ' + name)
    train = pd.read_csv(root/'data/processed/train.csv')
    validation = pd.read_csv(root/'data/processed/validation.csv')
    artists = pd.read_csv(raw/'artists.dat', sep='\t')
    events = pd.read_csv(raw/'user_taggedartists-timestamps.dat', sep='\t')
    tags = pd.read_csv(raw/'tags.dat', sep='\t', encoding='latin-1', usecols=['tagID'])
    if tags.tagID.duplicated().any():
        raise ValueError('Duplicate IDs in static tags.dat namespace')
    matrix, user_ids, artist_ids, user_index, item_index = indexed_matrix(train, artists)
    popularity = popularity_scores(matrix)
    data = {
        'train': train, 'validation': validation, 'matrix': matrix,
        'user_ids': user_ids, 'artist_ids': artist_ids,
        'user_index': user_index, 'item_index': item_index,
        'popularity': popularity, 'warm': popularity > 0, 'seen': matrix.toarray().astype(bool),
        'users': train.userID.map(user_index).to_numpy(dtype=np.int64),
        'items': train.artistID.map(item_index).to_numpy(dtype=np.int64),
    }
    return data, events, np.sort(tags.tagID.to_numpy(dtype=np.int64)), input_hashes


def policy_data(data, banks, policy):
    """Use the same model shape and namespace for the matched control."""
    user_folds = banks['user_folds']
    if policy == 'global':
        user_folds = np.zeros_like(user_folds)
    return {**data, 'feature_ids': banks[policy + '_ids'],
            'feature_weights': banks[policy + '_weights'],
            'vocabulary': banks['namespace'], 'user_folds': user_folds}


def _output_directory(root, output):
    output = Path(output).resolve()
    if output == root or root in output.parents:
        allowed = root/'outputs/content_feature_audit'
        if output != allowed and allowed not in output.parents:
            raise ValueError('Repository output must be inside outputs/content_feature_audit')
    return output


def _summarize(output, results):
    frame = pd.DataFrame(results)
    frame.to_csv(output/'validation_runs.csv', index=False)
    metrics = [name for name in frame if '@' in name]
    frame.groupby('feature_policy')[metrics].agg(['mean', 'std']).to_csv(output/'validation_summary.csv')
    pairs = frame.pivot(index='seed', columns='feature_policy', values=metrics)
    differences = pd.DataFrame({'seed': SEEDS})
    for metric in metrics:
        differences[metric + '_crossfit_minus_global'] = (
            pairs[(metric, 'crossfit')] - pairs[(metric, 'global')]).reindex(SEEDS).to_numpy()
    differences.to_csv(output/'paired_seed_differences.csv', index=False)
    user_pairs = []
    for seed in SEEDS:
        run = f"{CONFIG['family']}_lr{CONFIG['lr']:g}_seed{seed}"
        baseline = pd.read_csv(output/'global'/f'{run}_users.csv')
        corrected = pd.read_csv(output/'crossfit'/f'{run}_users.csv')
        if set(baseline.userID) != set(corrected.userID):
            raise ValueError('Matched policies evaluated different users')
        merged = baseline.merge(corrected, on='userID', suffixes=('_global', '_crossfit'),
                                validate='one_to_one')
        rows = merged[['userID']].copy()
        rows.insert(0, 'seed', seed)
        for metric in [name for name in baseline if '@' in name]:
            rows[metric + '_crossfit_minus_global'] = merged[metric + '_crossfit'] - merged[metric + '_global']
        user_pairs.append(rows)
    pd.concat(user_pairs, ignore_index=True).to_csv(output/'paired_user_differences.csv', index=False)


def _save_manifest(output, input_hashes, source_hashes):
    excluded = {'artifact_manifest.json', 'replay.json', 'status.json'}
    artifacts = {p.relative_to(output).as_posix(): sha256(p)
                 for p in sorted(output.rglob('*')) if p.is_file() and p.name not in excluded}
    save_json(output/'artifact_manifest.json', {
        'input_sha256': input_hashes,
        'source_sha256': source_hashes,
        'artifacts_sha256': artifacts,
    })


def replay_checkpoints(root, output):
    """Re-score only validation, verifying sources, inputs and saved artifacts."""
    root = Path(root).resolve()
    output = _output_directory(root, output)
    data, _, _, inputs = load_inputs(root)
    manifest = json.loads((output/'artifact_manifest.json').read_text(encoding='utf-8'))
    if inputs != manifest['input_sha256']:
        raise ValueError('Audit input hashes differ from saved experiment')
    for name, expected in manifest['source_sha256'].items():
        if sha256(root/name) != expected:
            raise ValueError('Audit source changed: ' + name)
    for name, expected in manifest['artifacts_sha256'].items():
        path = (output/name).resolve()
        if output not in path.parents or sha256(path) != expected:
            raise ValueError('Audit artifact changed: ' + name)
    with np.load(output/'feature_banks.npz', allow_pickle=False) as stored:
        banks = {name: stored[name].copy() for name in stored.files}
    if not np.array_equal(data['user_ids'], banks['user_ids']) or not np.array_equal(data['artist_ids'], banks['artist_ids']):
        raise ValueError('Saved bank catalogue or user ordering differs')
    protocol = json.loads((output/'protocol.json').read_text(encoding='utf-8'))
    checks = []
    torch.set_num_threads(protocol['threads'])
    for policy in ('global', 'crossfit'):
        current = policy_data(data, banks, policy)
        for seed in protocol['seeds']:
            run = f"{CONFIG['family']}_lr{CONFIG['lr']:g}_seed{seed}"
            saved = json.loads((output/policy/f'{run}_result.json').read_text(encoding='utf-8'))
            model = make_model(CONFIG, current)
            with np.load(output/policy/f'{run}_weights.npz', allow_pickle=False) as stored:
                model.load_state_dict({name: torch.from_numpy(stored[name].copy())
                                       for name in stored.files}, strict=True)
            expected_buffers = {'feature_ids': current['feature_ids'].reshape(model.feature_ids.shape),
                                'feature_weights': current['feature_weights'].reshape(model.feature_weights.shape),
                                'user_folds': current['user_folds']}
            for name, expected in expected_buffers.items():
                if not np.array_equal(getattr(model, name).numpy(), expected):
                    raise ValueError('Checkpoint content bank differs: ' + policy + '/' + name)
            metrics, detail = evaluate_model(model, current)
            errors = {name: abs(metrics[name] - saved[name]) for name in metrics if '@' in name}
            saved_users = pd.read_csv(output/policy/f'{run}_users.csv')
            pd.testing.assert_frame_equal(detail, saved_users, check_exact=False, atol=1e-7, rtol=0)
            if not errors or not np.isfinite(list(errors.values())).all() or max(errors.values()) > 1e-7:
                raise ValueError('Validation checkpoint replay mismatch: ' + policy + '/' + run)
            checks.append({'policy': policy, 'seed': seed,
                           'max_abs_metric_error': max(errors.values()), 'passed': True})
            print('Replayed', policy, seed, metrics['ndcg@10'], flush=True)
    if {(check['policy'], check['seed']) for check in checks} != {
            (policy, seed) for policy in ('global', 'crossfit') for seed in SEEDS} or len(checks) != 6:
        raise ValueError('Replay requires exactly six distinct planned checkpoints')
    report = {'status': 'passed', 'checks': checks, 'checkpoint_count': len(checks),
              'test_labels_read': False, 'test_metrics_evaluated': False}
    save_json(output/'replay.json', report)
    return report


def run_audit(root, output_dir=None, run_validation=False, replay_only=False):
    root = Path(root).resolve()
    output = _output_directory(root, output_dir or root/'outputs/content_feature_audit')
    if run_validation and replay_only:
        raise ValueError('Choose validation training or checkpoint replay')
    if replay_only:
        if not (output/'artifact_manifest.json').is_file():
            raise FileNotFoundError('Checkpoint replay requires a completed artifact manifest')
        save_json(output/'replay.json', {'status': 'running', 'test_labels_read': False,
                                        'test_metrics_evaluated': False})
        try:
            return replay_checkpoints(root, output)
        except Exception as error:
            save_json(output/'replay.json', {'status': 'failed', 'error_type': type(error).__name__,
                                            'error': str(error), 'test_labels_read': False,
                                            'test_metrics_evaluated': False})
            raise
    if any((output/name).exists() for name in ('global', 'crossfit', 'artifact_manifest.json')):
        raise FileExistsError('Existing validation artifacts require --replay-only or a new output directory')
    source_hashes = {}
    if run_validation:
        source_hashes = {name: sha256(root/name) for name in SOURCE_FILES}
    data, events, namespace, input_hashes = load_inputs(root)
    banks, bank_audits, vocabularies = build_feature_banks(
        events, data['train'], data['artist_ids'], namespace)
    audit = audit_self_contribution(events, data['train'], data['artist_ids'], banks, vocabularies['global'][0])
    output.mkdir(parents=True, exist_ok=True)
    save_json(output/'audit.json', audit)
    save_json(output/'bank_audits.json', bank_audits)
    save_json(output/'local_vocabularies.json', vocabularies)
    save_json(output/'static_tag_namespace.json', namespace.tolist())
    save_json(output/'input_hashes.json', input_hashes)
    pd.DataFrame({'userID': banks['user_ids'], 'fold': banks['user_folds']}).to_csv(
        output/'user_folds.csv', index=False)
    np.savez_compressed(output/'feature_banks.npz', **banks)
    protocol = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'baseline_commit': '5f0be782e6b50ea83efaf4e9ccbe954effb75b90',
        'config': CONFIG, 'seeds': list(SEEDS), 'fold_count': 5, 'fold_seed': 42,
        'fold_assignment': 'sorted user IDs, NumPy default_rng(42) permutation, balanced modulo 5',
        'namespace': 'all sorted numeric tagID values from static tags.dat; coordinate identity only',
        'policies': ['global', 'crossfit'], 'feature_limits': {'min_df': 2, 'max_features': 2000, 'max_tags': 32},
        'exclusion': 'exclude every event authored by the receiving user fold before local vocabulary, TF, DF, IDF and top32',
        'idf_documents': 'all train-observed artists, identical in both policies',
        'scoring': 'each user uses their own bank for positive, negative and full-catalogue validation scores',
        'max_epochs': 24, 'validation_every': 4, 'patience_evaluations': 3,
        'selection': 'validation macro NDCG@10, minimum improvement 1e-5; no new hyperparameter search',
        'optimizer': 'Adam coupled weight_decay=1e-4', 'batch_size': 1024,
        'negative_sampling': 'one uniform train-warm unseen-in-train item per positive each epoch',
        'device': 'cpu', 'threads': 4, 'torch': torch.__version__, 'numpy': np.__version__,
        'python': sys.version, 'platform': platform.platform(),
        'run_validation_requested': run_validation,
        'historical_comparability': 'Both new policies share the full static embedding shape. This differs from legacy 2000-tag model initialization and parameter count.',
        'interpretation_limit': 'Provider removal and feature coverage also change. Shared model parameters permit indirect cross-fold learning.',
        'test_labels_read': False, 'test_metrics_evaluated': False,
    }
    save_json(output/'protocol.json', protocol)
    status = {'stage': 'audit_complete', 'training_runs': 0, 'test_labels_read': False,
              'test_metrics_evaluated': False}
    if not run_validation:
        save_json(output/'status.json', status)
        return audit
    torch.set_num_threads(protocol['threads'])
    results = []
    status.update(stage='validation_running')
    save_json(output/'status.json', status)
    try:
        for policy in protocol['policies']:
            destination = output/policy
            destination.mkdir(parents=True)
            current = policy_data(data, banks, policy)
            for seed in SEEDS:
                result = train_config({**CONFIG, 'feature_policy': policy}, current, destination,
                                      seed=seed, max_epochs=24, every=4, patience=3)
                results.append(result)
                status.update(training_runs=len(results))
                save_json(output/'status.json', status)
        _summarize(output, results)
        if len({result['parameters'] for result in results}) != 1:
            raise ValueError('Matched policies have different parameter counts')
        if {name: sha256(root/name) for name in INPUT_FILES} != input_hashes:
            raise ValueError('Audit input changed during validation training')
        if {name: sha256(root/name) for name in SOURCE_FILES} != source_hashes:
            raise ValueError('Audit source changed during validation training')
        _save_manifest(output, input_hashes, source_hashes)
        replay = replay_checkpoints(root, output)
        status.update(stage='validation_complete', replay_passed=replay['status'] == 'passed')
    except Exception as error:
        status.update(stage='validation_failed', error_type=type(error).__name__, error=str(error))
        save_json(output/'status.json', status)
        raise
    save_json(output/'status.json', status)
    return pd.DataFrame(results)
