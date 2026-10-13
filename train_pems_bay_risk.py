"""Frozen PEMS-BAY risk training/selection; no calibration or evaluation reads."""
from __future__ import annotations
import contextlib
from datetime import datetime, timezone
from pathlib import Path
import time
from unittest.mock import patch
import h5py
import numpy as np
import torch
from freeze_external_validation import sha256
from verify_pems_bay_binding import main as verify_binding
from train_pems_bay_forecaster import read_json, write_json, load_model
from pems_bay_reader import SealedPartitionReader
from pems_bay_windows import contiguous_origins, calendar_features
from congestion_alert_pilot import SCENARIOS, infer, labels, outage_mask
from congestion_risk_pilot import features, score, sigmoid, RiskModel
from congestion_onset_pilot import onset_targets
from congestion_solver_audit import fit_linear
from congestion_density_pilot import fit_uniform

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT/'outputs/pems-bay-risk'
FORE = ROOT/'outputs/pems-bay-forecaster'
METHODS = ('dense_speed', 'global_calibration', 'group_calibration', 'state_logistic',
           'persistent_mlp', 'onset_linear', 'onset_mlp')
SEEDS = (202, 203, 204)
PARTITIONS = ('risk_train', 'risk_validation')


def verify_forecaster():
    verify_binding()
    for file in ('forecast_implementation_freeze.json', 'config.json'):
        for name, digest in read_json(FORE/file)['source_sha256'].items():
            if sha256(ROOT/name) != digest:
                raise ValueError(f'Forecaster source changed: {name}')
    for name, digest in read_json(FORE/'results.json')['artifact_sha256'].items():
        if sha256(FORE/name) != digest:
            raise ValueError(f'Forecaster artifact changed: {name}')
    audit = read_json(FORE/'audit.json')
    if audit.get('audit_status') != 'passed':
        raise ValueError('Forecaster audit not passed')


@contextlib.contextmanager
def guard_risk_reads(reader, trace):
    original = h5py.Dataset.__getitem__
    allowed = {reader.bounds[name]: name for name in PARTITIONS}
    columns = sorted(reader.columns.tolist())
    def guarded(dataset, key, *args, **kwargs):
        if dataset.name == '/speed/block0_values':
            if not isinstance(key, tuple) or len(key) != 2 or not isinstance(key[0], slice):
                raise PermissionError('Risk stage cannot read an unbounded speed array')
            rows = key[0]
            bounds = (rows.start, rows.stop)
            if bounds not in allowed or rows.step not in (None, 1) or np.asarray(key[1]).tolist() != columns:
                raise PermissionError('Risk stage may only read its two full selected-station partitions')
            trace.append({'partition': allowed[bounds], 'row_start': rows.start,
                          'row_stop_exclusive': rows.stop, 'columns': columns})
        return original(dataset, key, *args, **kwargs)
    with patch.object(h5py.Dataset, '__getitem__', guarded):
        yield


def prepare(reader, partition, models, statistics, groups):
    begun = time.perf_counter()
    raw64, timestamps = reader.read_partition(partition)
    raw = raw64.astype(np.float32)
    valid = (np.isfinite(raw64) & (raw64 > 0)).astype(np.float32)
    mean, scale, adjacency = [statistics[k] for k in ('mean', 'scale', 'adjacency')]
    z = np.where(valid > 0, (raw-mean)/scale, 0).astype(np.float32)
    origins = contiguous_origins(timestamps, 12, 3)
    context = calendar_features(timestamps)
    days = (timestamps.astype('datetime64[D]')-reader.d0).astype(int)
    truth, eligible, _ = labels(raw64, valid, origins, np.ones(len(origins), bool))
    onset, common = onset_targets((raw64, z, valid, timestamps), origins, 0, len(raw))
    assert not np.any(common & ~eligible)
    saved = {'origins': origins, 'timestamps': timestamps, 'truth': truth, 'eligible': eligible,
             'onset': onset, 'common': common, 'day_indices': days[origins]}
    for scenario in SCENARIOS:
        mask, fault = outage_mask(valid, days, scenario, groups)
        predictions = infer(models, z, mask, origins, context, mean, scale)
        state, single, group, names = features(raw, mask, origins, context, adjacency, predictions)
        saved.update({f'{scenario}_predictions': predictions, f'{scenario}_state': state,
                      f'{scenario}_single': single, f'{scenario}_group': group,
                      f'{scenario}_fault': fault[origins-1]})
        print(f'{partition} {scenario} prepared, seconds={time.perf_counter()-begun:.1f}', flush=True)
    summary = {'origins': len(origins), 'candidate_origins_before_time_gap_filter': len(raw)-14,
               'unique_three_reading_valid_points': int(eligible.sum()),
               'unique_common_six_reading_points': int(common.sum()), 'pooled_rows': int(common.sum())*3,
               'unique_persistent_positives': int((truth & common).sum()),
               'unique_onset_positives': int(onset.sum()), 'excluded_from_supervision': int((~common).sum()),
               'first_timestamp': str(timestamps[0]), 'last_timestamp': str(timestamps[-1]),
               'preparation_seconds': time.perf_counter()-begun}
    return saved, names, summary


def pooled(saved, feature):
    return np.concatenate([saved[f'{s}_{feature}'][saved['common']] for s in SCENARIOS])


def load_risk(path):
    saved = torch.load(path, weights_only=True)
    model = RiskModel(saved['dimension'], saved['nonlinear'])
    model.load_state_dict(saved['state_dict']); model.eval()
    return model


def scores(saved, fitted, norm):
    result = {}
    for s in SCENARIOS:
        shape = saved['truth'].shape
        x = ((saved[f'{s}_state'].reshape(-1, 51)-norm['mean'])/norm['std']).astype(np.float32)
        single = ((saved[f'{s}_single'].reshape(-1, 1)-norm['single_mean'])/norm['single_std']).astype(np.float32)
        result[f'dense_speed__{s}'] = saved[f'{s}_single']*70
        glob = sigmoid(score(fitted['global_calibration'][0], single)).reshape(shape)
        result[f'global_calibration__{s}'] = glob
        grouped = glob.copy()
        for g, models in fitted['group_calibration'].items():
            choose = saved[f'{s}_group'].reshape(-1) == int(g)
            grouped.reshape(-1)[choose] = sigmoid(score(models[0], single[choose]))
        result[f'group_calibration__{s}'] = grouped
        for method in ('state_logistic', 'persistent_mlp', 'onset_linear', 'onset_mlp'):
            result[f'{method}__{s}'] = np.mean([sigmoid(score(m, x)) for m in fitted[method]], axis=0).reshape(shape)
    return result


def probability_metrics(probability, target):
    p = probability.astype(np.float64)
    q = np.clip(p, 1e-12, 1-1e-12)
    return {'bce': float(np.mean(-target*np.log(q)-(1-target)*np.log1p(-q))),
            'brier': float(np.mean((p-target)**2)), 'rows': len(target), 'positives': int(target.sum())}


def main():
    verify_forecaster()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    sources = ['train_pems_bay_risk.py', 'audit_pems_bay_risk.py', 'docs/pems_bay_risk_protocol.zh-CN.md',
               'configs/pems_bay_validation_v1.json', 'pems_bay_reader.py', 'pems_bay_windows.py',
               'congestion_alert_pilot.py', 'congestion_risk_pilot.py', 'congestion_onset_pilot.py',
               'congestion_solver_audit.py', 'congestion_density_pilot.py', 'train_pems_bay_forecaster.py',
               'outputs/pems-bay-forecaster/forecast_implementation_freeze.json',
               'outputs/pems-bay-binding/forecast_training_statistics.npz', 'outputs/pems-bay-binding/fault_groups.json']
    sources += [f'outputs/pems-bay-forecaster/model_tcn_seed{seed}.pt' for seed in (42, 43, 44)]
    config = {'stage': 'risk_train_and_risk_validation_only', 'methods': list(METHODS),
              'seeds': list(SEEDS), 'source_sha256': {name: sha256(ROOT/name) for name in sources},
              'torch_version': torch.__version__, 'numpy_version': np.__version__, 'device': 'cpu',
              'calibration_executed': False, 'evaluation_values_read': False}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config_path = OUTPUT/'config.json'
    if config_path.exists():
        if read_json(config_path) != config: raise ValueError('Risk run configuration changed')
    else: write_json(config_path, config)
    if (OUTPUT/'results.json').exists():
        for name, digest in read_json(OUTPUT/'results.json')['artifact_sha256'].items():
            assert sha256(OUTPUT/name) == digest, name
        print('Existing completed risk training verified; no rerun'); return
    begun = time.perf_counter()
    design = read_json(ROOT/'configs/pems_bay_validation_v1.json')
    reader = SealedPartitionReader(ROOT/design['dataset']['local_path'], design)
    with np.load(ROOT/'outputs/pems-bay-binding/forecast_training_statistics.npz') as f:
        statistics = {k: f[k].copy() for k in ('mean', 'scale', 'adjacency')}
    models = [load_model(seed, statistics['adjacency'], FORE/f'model_tcn_seed{seed}.pt') for seed in (42, 43, 44)]
    groups = read_json(ROOT/'outputs/pems-bay-binding/fault_groups.json')
    saved, summaries, interceptions = {}, {}, []
    with guard_risk_reads(reader, interceptions):
        for partition in PARTITIONS:
            path = OUTPUT/f'{partition}.npz'; log_path = OUTPUT/f'{partition}.json'
            if path.exists() and log_path.exists():
                info = read_json(log_path)
                assert info['cache_sha256'] == sha256(path) and info['config_sha256'] == sha256(config_path)
                with np.load(path) as f: saved[partition] = {k: f[k] for k in f.files}
                summaries[partition] = info['summary']
            elif path.exists() or log_path.exists():
                raise ValueError('Incomplete risk cache preserved; no overwrite')
            else:
                saved[partition], names, summaries[partition] = prepare(reader, partition, models, statistics, groups)
                np.savez_compressed(path, **saved[partition])
                write_json(log_path, {'summary': summaries[partition], 'feature_names': names,
                                     'cache_sha256': sha256(path), 'config_sha256': sha256(config_path),
                                     'reader': reader.access_log[-1], 'interception': interceptions[-1]})
        write_json(OUTPUT/'access_log.json', {'reader_this_invocation': reader.access_log,
                   'runtime_this_invocation': interceptions,
                   'preparation_reads': [read_json(OUTPUT/f'{p}.json')['reader'] for p in PARTITIONS],
                   'calibration_values_read': False, 'evaluation_values_read': False})
        tr, va = saved['risk_train'], saved['risk_validation']
        unscaled = pooled(tr, 'state'); single_raw = pooled(tr, 'single')[:, None]
        norm = {'mean': unscaled.mean(axis=0), 'std': np.maximum(unscaled.std(axis=0), 1e-3),
                'single_mean': single_raw.mean(axis=0), 'single_std': np.maximum(single_raw.std(axis=0), 1e-3)}
        np.savez_compressed(OUTPUT/'normalization.npz', **norm)
        x = ((unscaled-norm['mean'])/norm['std']).astype(np.float32); del unscaled
        xv = ((pooled(va, 'state')-norm['mean'])/norm['std']).astype(np.float32)
        single = ((single_raw-norm['single_mean'])/norm['single_std']).astype(np.float32)
        sv = ((pooled(va, 'single')[:, None]-norm['single_mean'])/norm['single_std']).astype(np.float32)
        targets = {p: {k: np.tile(v[k][v['common']], 3).astype(np.float32) for k in ('truth', 'onset')} for p, v in saved.items()}
        y, yv = targets['risk_train']['truth'], targets['risk_validation']['truth']
        gt, gv = pooled(tr, 'group'), pooled(va, 'group')
        fitted = {'group_calibration': {}, 'persistent_mlp': [], 'onset_mlp': []}; logs = {}
        def fit_or_load(method, a, b, av, bv, seed=None):
            name = f'{method}_seed{seed}' if seed is not None else method
            path, log_path = OUTPUT/f'{name}.pt', OUTPUT/f'{name}.json'
            if path.exists() and log_path.exists():
                rec = read_json(log_path)
                assert rec['weight_sha256'] == sha256(path) and rec['config_sha256'] == sha256(config_path)
                model = load_risk(path)
            elif path.exists() or log_path.exists(): raise ValueError('Incomplete risk weight preserved')
            else:
                started = time.perf_counter(); updates = [0]
                original = torch.optim.Adam.step
                def tracked(optimizer, *args, **kwargs):
                    value = original(optimizer, *args, **kwargs); updates[0] += 1; return value
                if seed is None: model, rec = fit_linear(a, b, av, bv, path)
                else:
                    with patch.object(torch.optim.Adam, 'step', tracked):
                        model, rec = fit_uniform(a, b, av, bv, seed, path)
                    assert updates[0] == 1350
                    rec['actual_optimizer_updates'] = updates[0]
                rec.update({'weight_sha256': sha256(path), 'config_sha256': sha256(config_path),
                            'training_seconds': time.perf_counter()-started})
                write_json(log_path, rec)
            if seed is None:
                chosen = next(r for r in rec['logs'] if r['alpha'] == rec['selected_alpha'])
                if chosen['max_abs_gradient'] > 1e-4:
                    raise ValueError(f'Selected linear baseline not converged: {name}')
            print(f'{name}: selected BCE={rec["validation_bce"]:.6f}, seconds={rec["training_seconds"]:.1f}', flush=True)
            return model, rec
        model, logs['global_calibration'] = fit_or_load('global_calibration', single, y, sv, yv)
        fitted['global_calibration'] = [model]; logs['group_calibration'] = {}
        for g in (0, 1, 2):
            a, b = gt == g, gv == g
            reasons = (['training_rows_below_100'] if a.sum() < 100 else [])
            if len(np.unique(y[a])) < 2: reasons.append('one_training_class')
            if not b.any(): reasons.append('empty_validation')
            if reasons:
                logs['group_calibration'][str(g)] = {'fallback': 'global_calibration', 'reasons': reasons,
                    'training_rows': int(a.sum()), 'validation_rows': int(b.sum())}
            else:
                model, rec = fit_or_load(f'group{g}', single[a], y[a], sv[b], yv[b])
                fitted['group_calibration'][str(g)] = [model]; logs['group_calibration'][str(g)] = rec
        for method, objective in (('state_logistic', 'truth'), ('onset_linear', 'onset')):
            model, logs[method] = fit_or_load(method, x, targets['risk_train'][objective], xv, targets['risk_validation'][objective])
            fitted[method] = [model]
        for method, objective in (('persistent_mlp', 'truth'), ('onset_mlp', 'onset')):
            logs[method] = []
            for seed in SEEDS:
                model, rec = fit_or_load(method, x, targets['risk_train'][objective], xv, targets['risk_validation'][objective], seed)
                fitted[method].append(model); logs[method].append(rec)
        write_json(OUTPUT/'model_selection.json', logs)
        validation_metrics = {'persistent_target_Y': {}, 'onset_target_Z': {}}
        for partition, part in saved.items():
            sc = scores(part, fitted, norm)
            np.savez_compressed(OUTPUT/f'scores_{partition}.npz', **sc)
            if partition == 'risk_validation':
                for method in METHODS[1:]:
                    objective = 'onset' if method.startswith('onset_') else 'truth'
                    pool_prob = np.concatenate([sc[f'{method}__{s}'][part['common']] for s in SCENARIOS])
                    panel = 'onset_target_Z' if objective == 'onset' else 'persistent_target_Y'
                    validation_metrics[panel][method] = probability_metrics(pool_prob, targets[partition][objective])
    for name, digest in config['source_sha256'].items(): assert sha256(ROOT/name) == digest, name
    artifacts = sorted(p.name for p in OUTPUT.iterdir() if p.is_file() and p.name != 'results.json')
    result = {'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'stage': config['stage'],
              'partitions': summaries, 'validation': validation_metrics,
              'elapsed_seconds': time.perf_counter()-begun, 'risk_training_executed': True,
              'calibration_executed': False, 'evaluation_values_read': False,
              'held_out_evaluation_executed': False,
              'artifact_sha256': {name: sha256(OUTPUT/name) for name in artifacts}}
    write_json(OUTPUT/'results.json', result)
    print(result['validation'], flush=True)


if __name__ == '__main__': main()
