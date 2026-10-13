"""Frozen traffic predictions -> observable congestion risk -> calibrated alerts."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from torch import nn

from recovery_pilot import ROOT, write_json
from state_value_pilot import read_data
from fresh_holdout_pilot import MaskedGraphTCN
from multifault_pilot import sha256
from congestion_alert_pilot import SCENARIOS, infer, labels, outage_mask, event_stats, point_stats

BASE = ROOT/'outputs/congestion-alert-pilot-v2'
PROB_THRESHOLDS = [.01, .02, .03, .05, .075, .10, .15, .20, .25, .30, .40, .50, .60, .70, .80, .90, 1.0001]
METHODS = ('dense_speed', 'global_calibration', 'group_calibration', 'state_logistic', 'risk_mlp')
BASELINES = METHODS[:-1]
CHECK_EPOCHS = (5, 10, 20, 30)


class RiskModel(nn.Module):
    def __init__(self, dimension, nonlinear=False):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dimension, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)) if nonlinear else nn.Linear(dimension, 1)

    def forward(self, x):
        return self.layers(x).squeeze(-1)


def features(raw, mask, origins, context, adjacency, predictions):
    """No future raw speed, target coverage, fault identity, or hidden values."""
    history = origins[:, None]+np.arange(-12, 0)[None]
    m = mask[history]
    x = np.where(m > 0, raw[history], 0)
    count = m.sum(axis=1)
    denominator = np.maximum(count, 1)
    avg = x.sum(axis=1)/denominator
    std = np.sqrt(np.maximum((x*x).sum(axis=1)/denominator-avg*avg, 0))
    index = np.where(m > 0, np.arange(12)[None, :, None], -1).max(axis=1)
    first = np.where(m > 0, np.arange(12)[None, :, None], 12).min(axis=1)
    rows, nodes = np.arange(len(origins))[:, None], np.arange(raw.shape[1])[None]
    last = x[rows, np.maximum(index, 0), nodes]
    first_value = x[rows, np.minimum(first, 11), nodes]
    trend = (last-first_value)/np.maximum(index-first, 1)
    own = np.stack([last/70, avg/70, trend/20, std/20, count/12,
                    (11-index)/12, ((x <= 30)*(m > 0)).sum(axis=1)/denominator], axis=-1)
    graph = adjacency.copy(); np.fill_diagonal(graph, 0)
    graph /= np.maximum(graph.sum(axis=1, keepdims=True), 1e-6)
    weights = graph[None]*(count > 0)[:, None]
    available = weights.sum(axis=-1)
    summary = np.stack([last/70, avg/70, trend/20, std/20], axis=-1)
    neighbor = np.einsum('bij,bjk->bik', weights, summary)/np.maximum(available[:, :, None], 1e-6)
    neighbor = np.concatenate([neighbor, available[:, :, None]], axis=-1)
    forecast = predictions.mean(axis=0).transpose(0, 2, 1)
    spread = predictions.std(axis=0).transpose(0, 2, 1)
    fore = np.concatenate([forecast/70, spread/20, forecast.max(axis=-1, keepdims=True)/70,
                           np.ptp(forecast, axis=-1)[:, :, None]/20], axis=-1)
    b, n = count.shape
    calendar = np.broadcast_to(context[origins, None, :], (b, n, 4))
    identity = np.broadcast_to(np.eye(n, dtype=np.float32)[None], (b, n, n))
    network = np.broadcast_to(m.mean(axis=(1, 2))[:, None, None], (b, n, 1))
    flags = np.stack([count == 0, (count > 0) & (count < 12)], axis=-1)
    state = np.concatenate([fore, own, neighbor, calendar, identity, network, flags], axis=-1).astype(np.float32)
    group = np.where(count == 0, 0, np.where(count == 12, 2, 1)).astype(np.int8)
    single = (-forecast.max(axis=-1)/70).astype(np.float32)
    names = ['forecast5', 'forecast10', 'forecast15', 'seed_std5', 'seed_std10', 'seed_std15', 'forecast_max', 'forecast_range',
             'own_last', 'own_mean', 'own_trend', 'own_std', 'own_coverage', 'own_age', 'own_low_fraction',
             'neighbor_last', 'neighbor_mean', 'neighbor_trend', 'neighbor_std', 'neighbor_coverage']
    names += [f'calendar{i}' for i in range(4)]+[f'node{i}' for i in range(n)]+['network_coverage', 'own_unavailable', 'own_partial']
    assert state.shape[-1] == len(names) and np.isfinite(state).all()
    return state, single, group, names


def sigmoid(a):
    return 1/(1+np.exp(-np.clip(a, -50, 50)))


def score(model, x, batch=8192):
    outputs = []
    model.eval()
    with torch.no_grad():
        for at in range(0, len(x), batch):
            outputs.append(model(torch.tensor(x[at:at+batch], dtype=torch.float32)).numpy())
    return np.concatenate(outputs)


def fit(x, y, xv, yv, nonlinear, seed, path):
    torch.manual_seed(seed)
    model = RiskModel(x.shape[1], nonlinear)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001 if nonlinear else .01, weight_decay=.001)
    loss_fn = nn.BCEWithLogitsLoss()
    xt, yt = torch.tensor(x), torch.tensor(y, dtype=torch.float32)
    rng = torch.Generator().manual_seed(seed+17)
    logs, best, state, epoch_selected = [], float('inf'), None, None
    for epoch in range(1, 31):
        model.train()
        indices = torch.randperm(len(x), generator=rng)
        for ids in indices.split(4096):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xt[ids]), yt[ids])
            loss.backward(); optimizer.step()
        if epoch in CHECK_EPOCHS:
            logits = score(model, xv)
            val_loss = float(np.mean(np.logaddexp(0, logits)-yv*logits))
            logs.append({'epoch': epoch, 'validation_bce': val_loss})
            if val_loss < best:
                best, epoch_selected, state = val_loss, epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(state); model.eval()
    record = {'seed': seed, 'nonlinear': nonlinear, 'dimension': x.shape[1], 'selected_epoch': epoch_selected,
              'validation_bce': best, 'training_rows': len(x), 'validation_rows': len(xv), 'logs': logs}
    torch.save({'state_dict': state, **record}, path)
    return model, record


def online_alarm(scores, threshold, confirmation):
    active = scores >= threshold
    if confirmation == 1:
        return active
    result = np.zeros_like(active)
    result[1:] = active[1:] & active[:-1]
    return result


def operating(scores, saved, threshold, confirmation):
    alarm = online_alarm(scores, threshold, confirmation)
    stats, daily, columns = event_stats(saved['truth'], saved['eligible'], alarm, saved['day_ids'])
    stats['point_all'] = point_stats(saved['truth'], saved['eligible'], alarm)
    stats['point_fault_nodes'] = point_stats(saved['truth'], saved['eligible'] & saved['fault'], alarm)
    stats['point_other_nodes'] = point_stats(saved['truth'], saved['eligible'] & ~saved['fault'], alarm)
    return stats, daily


def summary_key(rows):
    warning = [r['warning_recall'] for r in rows]
    return (-float(np.mean(warning)), -float(np.min(warning)), float(np.mean([r['false_per_valid_station_day'] for r in rows])))


def choose(method, scores, saved, budgets):
    thresholds = [-float(v) for v in np.arange(15, 60.1, 2.5)]+[1000.] if method == 'dense_speed' else PROB_THRESHOLDS
    grid, candidates = [], []
    for i, threshold in enumerate(thresholds):
        for confirmation in (1, 2):
            rows = [operating(scores[s], saved[s], threshold, confirmation)[0] for s in SCENARIOS]
            feasible = all(row['false_per_valid_station_day'] <= budgets[s]+1e-12 for s, row in zip(SCENARIOS, rows))
            record = {'threshold': threshold, 'confirmation': confirmation, 'feasible': feasible,
                      'by_scenario': {s: {k: r[k] for k in ('warning_recall', 'false_per_valid_station_day', 'warning_lead_mean_minutes')} for s, r in zip(SCENARIOS, rows)}}
            grid.append(record)
            if feasible:
                candidates.append(((*summary_key(rows), confirmation, i), record))
    if not candidates:
        raise ValueError('No feasible operating point, including off rule')
    selected = min(candidates, key=lambda v: v[0])[1]
    return {'selected': selected, 'grid': grid}


def aggregate_ci(reference, candidate):
    # Same date indices across scenarios preserve pairing of shared truth/weather.
    rng = np.random.default_rng(6110)
    draws = rng.integers(0, reference.shape[1], size=(2000, reference.shape[1]))
    a = reference[:, draws, :].sum(axis=2)
    b = candidate[:, draws, :].sum(axis=2)
    wr = (b[:, :, 2]/np.maximum(b[:, :, 1], 1)-a[:, :, 2]/np.maximum(a[:, :, 1], 1)).mean(axis=0)
    fa = (b[:, :, 4]/b[:, :, 0]-a[:, :, 4]/a[:, :, 0]).mean(axis=0)
    return {'mean_warning_recall_difference_ci': np.quantile(wr, [.025, .975]).tolist(),
            'mean_false_rate_difference_ci': np.quantile(fa, [.025, .975]).tolist()}


def prepare_partition(partition, start, end, stride, data, models, output):
    raw, z, valid, ts, context, mean, scale, adjacency, day, groups = data
    source = BASE/f'predictions_{partition}.npz'
    cache = output/f'predictions_{partition}.npz'
    if source.exists():
        with np.load(source) as f:
            saved = {k: f[k] for k in f.files}
    elif cache.exists():
        with np.load(cache) as f:
            saved = {k: f[k] for k in f.files}
    else:
        origins = np.arange(start+12, end-2, stride)
        window = origins[:, None]+np.arange(-12, 3)[None]
        regular = (np.diff(ts[window], axis=1) == np.timedelta64(5, 'm')).all(axis=1)
        truth, eligible, future = labels(raw, valid, origins, regular)
        dates = np.unique(ts[origins].astype('datetime64[D]'))
        saved = {'origins': origins, 'truth': truth, 'eligible': eligible,
                 'day_ids': np.searchsorted(dates, ts[origins].astype('datetime64[D]')), 'dates': dates.astype(str)}
        for s in SCENARIOS:
            mask, fault = outage_mask(valid, day, s, groups)
            saved[f'predictions_{s}'] = infer(models, z, mask, origins, context, mean, scale)
            saved[f'fault_{s}'] = fault[origins-1]
            print(f'{partition} forecast {s}', flush=True)
        np.savez_compressed(cache, **saved)
    partition_data = {}
    for s in SCENARIOS:
        mask, fault = outage_mask(valid, day, s, groups)
        state, single, group, names = features(raw, mask, saved['origins'], context, adjacency, saved[f'predictions_{s}'])
        partition_data[s] = {'state': state, 'single': single, 'group': group,
                             'truth': saved['truth'], 'eligible': saved['eligible'], 'day_ids': saved['day_ids'],
                             'fault': saved[f'fault_{s}'], 'origins': saved['origins'], 'dates': saved['dates']}
    return partition_data, names


def pool(parts, feature):
    return np.concatenate([parts[s][feature][parts[s]['eligible']] for s in SCENARIOS])


def full_scores(parts, models, norm):
    mu, sigma, smu, sstd = norm
    global_model, group_models, logistic, mlps = models
    results = {method: {} for method in METHODS}
    for s in SCENARIOS:
        part = parts[s]; shape = part['truth'].shape
        x = ((part['state'].reshape(-1, len(mu))-mu)/sigma).astype(np.float32)
        single = ((part['single'].reshape(-1, 1)-smu)/sstd).astype(np.float32)
        groups = part['group'].reshape(-1)
        results['dense_speed'][s] = part['single']*70
        results['global_calibration'][s] = sigmoid(score(global_model, single)).reshape(shape)
        grouped = sigmoid(score(global_model, single))
        for g, model in group_models.items():
            rows = groups == int(g)
            if rows.any():
                grouped[rows] = sigmoid(score(model, single[rows]))
        results['group_calibration'][s] = grouped.reshape(shape)
        results['state_logistic'][s] = sigmoid(score(logistic, x)).reshape(shape)
        results['risk_mlp'][s] = np.mean([sigmoid(score(model, x)) for model in mlps], axis=0).reshape(shape)
    return results


def hidden_features_check(part, data):
    raw, z, valid, ts, context, mean, scale, adjacency, day, groups = data
    origins = part['group_outage']['origins'][:10]
    mask, _ = outage_mask(valid, day, 'group_outage', groups)
    # Forecasts are fixed to isolate state-feature leakage; original forecaster audited separately.
    predictions = np.zeros((3, len(origins), 3, raw.shape[1]), np.float32)
    a = features(raw, mask, origins, context, adjacency, predictions)
    modified = np.where(mask > 0, raw, -1e6)
    b = features(modified, mask, origins, context, adjacency, predictions)
    assert all(np.array_equal(x, y) for x, y in zip(a[:3], b[:3]))
    # Change all current/future raw data at one origin: features must remain identical.
    changed = raw.copy(); changed[origins[0]:] = 1e6
    c = features(changed, mask, origins[:1], context, adjacency, predictions[:, :1])
    assert all(np.array_equal(x[:1], y) for x, y in zip(a[:3], c[:3]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/congestion-risk-pilot')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(); output = args.output; output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    z, valid, scale, adjacency, ts, context, cuts = read_data()
    old = json.loads((ROOT/'outputs/recovery-pilot/config.json').read_text(encoding='utf-8'))
    mean = np.array(old['normalization_mean'], np.float32)
    with h5py.File(ROOT/'data/raw/metr-la.h5', 'r') as f:
        raw = f['df/block0_values'][:, old['raw_column_indices']].astype(np.float32)
    groups = json.loads((ROOT/'outputs/multifault-pilot/config.json').read_text(encoding='utf-8'))['groups']
    day = (ts.astype('datetime64[D]')-ts[0].astype('datetime64[D]')).astype(int)
    paths = [Path(__file__), ROOT/'docs/congestion_risk_protocol.zh-CN.md', ROOT/'congestion_alert_pilot.py',
             ROOT/'recovery_pilot.py', ROOT/'state_value_pilot.py', ROOT/'fresh_holdout_pilot.py', ROOT/'data/raw/metr-la.h5',
             ROOT/'outputs/recovery-pilot/config.json', ROOT/'outputs/multifault-pilot/config.json']
    paths += [ROOT/'outputs/fresh-holdout-pilot'/f'model_tcn_seed{s}.pt' for s in (42, 43, 44)]
    paths += [BASE/'results.json', BASE/'config.json', BASE/'predictions_calibration.npz', BASE/'predictions_evaluation.npz']
    config = {'date': '2026-10-10', 'exploratory': True, 'all_dates_previously_used': True,
              'partition_days': {'training': [42, 70], 'validation': [70, 77], 'calibration': [77, 91], 'evaluation': [91, 119]},
              'train_stride': 3, 'methods': METHODS, 'probability_thresholds': PROB_THRESHOLDS,
              'mlp_seeds': [202, 203, 204], 'epochs': 30, 'batch': 4096,
              'sha256': {str(p.relative_to(ROOT)): sha256(p) for p in paths}}
    config = json.loads(json.dumps(config))
    if (output/'config.json').exists():
        assert config == json.loads((output/'config.json').read_text(encoding='utf-8')), 'Use a new output directory if frozen inputs change'
    else:
        write_json(output/'config.json', config)
    if args.prepare_only:
        print(json.dumps({'config': str(output/'config.json'), 'methods': METHODS})); return
    if (output/'results.json').exists():
        raise ValueError('Existing run preserved. Use a new output directory.')
    models = []
    for seed in (42, 43, 44):
        m = MaskedGraphTCN(12, 3, adjacency)
        m.load_state_dict(torch.load(ROOT/'outputs/fresh-holdout-pilot'/f'model_tcn_seed{seed}.pt', weights_only=True)['state_dict'])
        m.eval(); models.append(m)
    data = raw, z, valid, ts, context, mean, scale, adjacency, day, groups
    at = lambda d: int(np.searchsorted(day, d))
    train, feature_names = prepare_partition('training', at(42), at(70), 3, data, models, output)
    val, _ = prepare_partition('validation', at(70), at(77), 1, data, models, output)
    hidden_features_check(train, data)
    x, y = pool(train, 'state'), pool(train, 'truth').astype(np.float32)
    xv, yv = pool(val, 'state'), pool(val, 'truth').astype(np.float32)
    single, sv = pool(train, 'single')[:, None], pool(val, 'single')[:, None]
    gt, gv = pool(train, 'group'), pool(val, 'group')
    mu, sigma = x.mean(axis=0), x.std(axis=0); sigma[sigma < 1e-5] = 1
    smu, sstd = single.mean(axis=0), single.std(axis=0); sstd[sstd < 1e-5] = 1
    np.savez_compressed(output/'normalization.npz', mean=mu, std=sigma, single_mean=smu, single_std=sstd)
    x, xv = ((x-mu)/sigma).astype(np.float32), ((xv-mu)/sigma).astype(np.float32)
    single, sv = ((single-smu)/sstd).astype(np.float32), ((sv-smu)/sstd).astype(np.float32)
    logs = {}
    global_model, logs['global_calibration'] = fit(single, y, sv, yv, False, 202, output/'global.pt')
    print('global calibration fitted', flush=True)
    group_models = {}
    logs['group_calibration'] = {}
    for g in (0, 1, 2):
        a, b = gt == g, gv == g
        if a.sum() < 100 or len(np.unique(y[a])) < 2 or not b.any():
            logs['group_calibration'][str(g)] = {'fallback': 'global', 'training_rows': int(a.sum())}
            continue
        group_models[str(g)], logs['group_calibration'][str(g)] = fit(single[a], y[a], sv[b], yv[b], False, 202, output/f'group{g}.pt')
    logistic, logs['state_logistic'] = fit(x, y, xv, yv, False, 202, output/'logistic.pt')
    print('state logistic fitted', flush=True)
    mlps = []
    logs['risk_mlp'] = []
    for seed in (202, 203, 204):
        m, log = fit(x, y, xv, yv, True, seed, output/f'mlp{seed}.pt')
        mlps.append(m); logs['risk_mlp'].append(log)
        print(f'risk MLP seed={seed} selected_epoch={log["selected_epoch"]} elapsed={time.perf_counter()-started:.1f}s', flush=True)
    write_json(output/'model_selection.json', logs)
    write_json(output/'features.json', feature_names)
    fitted = global_model, group_models, logistic, mlps
    norm = mu, sigma, smu, sstd
    del train, val, x, xv, single, sv
    calibration, _ = prepare_partition('calibration', at(77), at(91), 1, data, models, output)
    scores_cal = full_scores(calibration, fitted, norm)
    old_results = json.loads((BASE/'results.json').read_text(encoding='utf-8'))
    budgets = {s: old_results['calibration'][s]['v30_c1']['false_per_valid_station_day'] for s in SCENARIOS}
    selections = {method: choose(method, scores_cal[method], calibration, budgets) for method in METHODS}
    def rank(method):
        selected = selections[method]['selected']
        rows = list(selected['by_scenario'].values())
        return (*summary_key(rows), BASELINES.index(method))
    reference = min(BASELINES, key=rank)
    selection = {'reference': reference, 'candidate': 'risk_mlp', 'budgets': budgets, 'methods': selections}
    write_json(output/'selection.json', selection)
    np.savez_compressed(output/'scores_calibration.npz', **{f'{m}_{s}': a for m, parts in scores_cal.items() for s, a in parts.items()})
    print(f'calibration-selected reference={reference}; evaluation not computed yet', flush=True)
    evaluation, _ = prepare_partition('evaluation', at(91), at(119), 1, data, models, output)
    hidden_features_check(evaluation, data)
    scores_eval = full_scores(evaluation, fitted, norm)
    np.savez_compressed(output/'scores_evaluation.npz', **{f'{m}_{s}': a for m, parts in scores_eval.items() for s, a in parts.items()})
    summaries, daily_all = {}, {}
    for method in METHODS:
        summaries[method] = {}
        op = selections[method]['selected']
        for s in SCENARIOS:
            row, daily = operating(scores_eval[method][s], evaluation[s], op['threshold'], op['confirmation'])
            if method != 'dense_speed':
                e, truth, p = evaluation[s]['eligible'], evaluation[s]['truth'], scores_eval[method][s]
                target = truth[e].astype(np.float64)
                row['brier'] = float(np.mean((p[e]-target)**2))
                row['bce'] = float(np.mean(-target*np.log(np.clip(p[e], 1e-7, 1-1e-7))-(1-target)*np.log(np.clip(1-p[e], 1e-7, 1))))
            summaries[method][s] = row
            daily_all[f'{method}_{s}'] = daily
    ref_daily = np.stack([daily_all[f'{reference}_{s}'] for s in SCENARIOS])
    mlp_daily = np.stack([daily_all[f'risk_mlp_{s}'] for s in SCENARIOS])
    comparison = aggregate_ci(ref_daily, mlp_daily)
    comparison.update({'mean_warning_recall_difference': float(np.mean([summaries['risk_mlp'][s]['warning_recall']-summaries[reference][s]['warning_recall'] for s in SCENARIOS])),
                       'mean_false_rate_difference': float(np.mean([summaries['risk_mlp'][s]['false_per_valid_station_day']-summaries[reference][s]['false_per_valid_station_day'] for s in SCENARIOS]))})
    result = {'config': config, 'reference': reference, 'candidate': 'risk_mlp', 'selection': selection,
              'evaluation': summaries, 'comparison': comparison, 'model_selection': logs,
              'elapsed_seconds': time.perf_counter()-started, 'new_method_originality_confirmed': False}
    np.savez_compressed(output/'daily_metrics.npz', **daily_all)
    write_json(output/'results.json', result)
    audit = {'hidden_and_future_state_features_invariant': True, 'source_hashes_unchanged': all(sha256(ROOT/k) == v for k, v in config['sha256'].items()),
             'trained_forecasters': False, 'selection_saved_before_evaluation': True,
             'artifact_sha256': {p.name: sha256(p) for p in output.iterdir() if p.suffix in ('.npz', '.pt')},
             'audit_status': 'initial checks; independent recomputation pending'}
    assert audit['source_hashes_unchanged']
    write_json(output/'audit.json', audit)
    print(json.dumps({'seconds': result['elapsed_seconds'], 'reference': reference, 'comparison': comparison,
        'candidate_evaluation': {s: {k: summaries['risk_mlp'][s][k] for k in ('warning_recall', 'false_per_valid_station_day')} for s in SCENARIOS}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
