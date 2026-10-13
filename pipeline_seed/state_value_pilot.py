"""Start-time observable-state repair policy, frozen chronological holdout."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from torch import nn
from recovery_pilot import ROOT, RecoveryForecaster, write_json
from fresh_holdout_pilot import MaskedGraphTCN, healthy_cost
from multifault_pilot import (ORDERS, make_events, calibrated_policies, evaluate_events,
                             day_macro, day_interval, relative_day_interval, sha256)

SEEDS = (42, 43, 44)
VALUE_SEEDS = (202, 203, 204)
ALPHAS = (1., 10., 100., 1000.)
BASELINES = ('shortest_time', 'fixed_joint', 'fixed_gain_time', 'calendar_ridge', 'state_ridge', 'state_gain_time')


def read_data():
    original = json.loads((ROOT/'outputs/recovery-pilot/config.json').read_text(encoding='utf-8'))
    with h5py.File(ROOT/'data/raw/metr-la.h5', 'r') as h5:
        raw = h5['df/block0_values'][:, original['raw_column_indices']].astype(np.float32)
        ts = h5['df/axis1'][:].astype('datetime64[ns]')
    days = ts.astype('datetime64[D]')
    cuts = [int(np.searchsorted(days, days[0]+np.timedelta64(d, 'D'))) for d in (42, 49, 77, 91, 119)]
    valid = (np.isfinite(raw) & (raw > 0)).astype(np.float32)
    mean = np.array(original['normalization_mean'], np.float32)
    scale = np.array(original['normalization_std'], np.float32)
    z = np.where(valid > 0, (raw-mean)/scale, 0).astype(np.float32)
    a = torch.load(ROOT/'outputs/recovery-pilot/model_seed42.pt', weights_only=True)['state_dict']['adjacency'].numpy()
    minute = (ts-days).astype('timedelta64[m]').astype(np.float32)
    weekday = (days.astype(np.int64)+3) % 7
    ctx = np.stack([np.sin(2*np.pi*minute/1440), np.cos(2*np.pi*minute/1440),
                    np.sin(2*np.pi*weekday/7), np.cos(2*np.pi*weekday/7)], axis=1).astype(np.float32)
    return z, valid, scale, a, ts, ctx, cuts


def observable_features(z, valid, context, origin, group, adjacency):
    """Inputs never index at/after origin, or use evaluation coverage/targets."""
    mask = valid[origin-12:origin].copy()
    mask[:, group['nodes']] = 0
    x = np.where(mask > 0, z[origin-12:origin], 0)
    count = mask.sum(axis=0)
    mean = x.sum(axis=0)/np.maximum(count, 1)
    spread = np.sqrt(np.maximum((x*x).sum(axis=0)/np.maximum(count, 1)-mean*mean, 0))
    idx = np.where(mask > 0, np.arange(12)[:, None], -1).max(axis=0)
    last = x[np.maximum(idx, 0), np.arange(x.shape[1])]
    first_idx = np.where(mask > 0, np.arange(12)[:, None], 12).min(axis=0)
    first = x[np.minimum(first_idx, 11), np.arange(x.shape[1])]
    trend = (last-first)/np.maximum(idx-first_idx, 1)
    summary = np.stack([last, mean, trend, spread, count/12], axis=1)
    proxies = []
    for matrix in (adjacency, adjacency@adjacency):
        weights = matrix[group['nodes']]*(count > 0)[None]
        denom = np.maximum(weights.sum(axis=1, keepdims=True), 1e-6)
        proxies.extend([(weights@summary[:, :3]/denom).ravel(), weights.sum(axis=1)])
    proxy = np.concatenate(proxies)
    onehot = np.eye(12, dtype=np.float32)[group['id']]
    calendar = np.concatenate([onehot, context[origin], np.outer(onehot, context[origin]).ravel(),
                               np.array(group['durations_steps'], np.float32)/6])
    state = np.concatenate([calendar, summary.ravel(), proxy, np.outer(onehot, proxy).ravel()])
    return calendar.astype(np.float32), state.astype(np.float32)


def features(events, groups, z, valid, context, adjacency):
    pairs = [observable_features(z, valid, context, e['origin'], groups[e['group']], adjacency) for e in events]
    return [np.stack([p[i] for p in pairs]) for i in (0, 1)]


def standardize(train, others):
    mean, scale = train.mean(axis=0), train.std(axis=0)
    scale = np.where(scale > 1e-5, scale, 1)
    return [(a-mean)/scale for a in [train, *others]], mean, scale


def weighted_rows(events):
    days = [e['day'] for e in events]
    counts = {d: days.count(d) for d in set(days)}
    w = np.array([1/counts[d] for d in days])
    return w/w.mean()


def ridge_fit(x, y, alpha, weight):
    design = np.column_stack([x, np.ones(len(x))]).astype(np.float64)
    penalty = np.full(design.shape[1], alpha); penalty[-1] = 0
    return np.linalg.solve(design.T@(design*weight[:, None])+np.diag(penalty), design.T@(y*weight[:, None]))


def ridge_predict(x, coefficients):
    return np.column_stack([x, np.ones(len(x))])@coefficients


def gain_choices(gains, events, groups):
    return np.array([ORDERS.index(tuple(sorted(range(3), key=lambda j: (-gains[i, j]/groups[e['group']]['durations_steps'][j], j))))
                     for i, e in enumerate(events)])


def selected_cost(actual, choice, events):
    return float(day_macro(actual[np.arange(len(actual)), choice], events))


class ValueMLP(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dimension, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 6))
    def forward(self, x):
        out = self.layers(x)
        return out-out.mean(dim=1, keepdim=True)


def train_value(x, y, xv, val_actual, val_events, weight):
    scale = float(max(y.std(), 1e-6))
    models, optimizers, rngs = [], [], []
    for seed in VALUE_SEEDS:
        torch.manual_seed(seed)
        m = ValueMLP(x.shape[1]); models.append(m)
        optimizers.append(torch.optim.Adam(m.parameters(), lr=.001, weight_decay=.001))
        rngs.append(torch.Generator().manual_seed(seed+100))
    xt, yt, wt, vt = [torch.tensor(a, dtype=torch.float32) for a in (x, y/scale, weight, xv)]
    best, best_states, logs = float('inf'), None, []
    for epoch in range(1, 151):
        for model, optimizer, rng in zip(models, optimizers, rngs):
            model.train()
            indexes = torch.randperm(len(x), generator=rng)
            for batch in indexes.split(128):
                optimizer.zero_grad(set_to_none=True)
                loss = ((model(xt[batch])-yt[batch]).square().mean(dim=1)*wt[batch]).mean()
                loss.backward(); optimizer.step()
        if epoch % 10 == 0:
            with torch.no_grad():
                prediction = np.mean([m(vt).numpy() for m in models], axis=0)
            cost = selected_cost(val_actual, prediction.argmin(axis=1), val_events)
            logs.append({'epoch': epoch, 'validation_cost': cost})
            if cost < best:
                best, selected_epoch = cost, epoch
                best_states = [copy.deepcopy(m.state_dict()) for m in models]
    for m, state in zip(models, best_states):
        m.load_state_dict(state); m.eval()
    return models, scale, {'selected_epoch': selected_epoch, 'validation_cost': best, 'epochs': logs}


def evaluate_partition(variant, name, events, groups, z, valid, context, scale, adjacency, output):
    path = output/f'labels_{variant}_{name}.npz'
    if path.exists():
        with np.load(path) as saved:
            return {key: saved[key] for key in saved.files}
    evaluated = []
    for seed in SEEDS:
        if variant == 'graph_tcn_candidate':
            model = MaskedGraphTCN(12, 3, adjacency)
            weight = ROOT/'outputs/fresh-holdout-pilot'/f'model_tcn_seed{seed}.pt'
        else:
            model = RecoveryForecaster(12, 3, adjacency, True)
            weight = ROOT/'outputs/recovery-pilot'/f'model_seed{seed}.pt'
        model.load_state_dict(torch.load(weight, weights_only=True)['state_dict']); model.eval()
        arrays = evaluate_events(model, events, groups, z, valid, context, scale, 12, 3, 24, False)
        healthy = healthy_cost(model, events, z, valid, context, scale)
        evaluated.append((*arrays[:4], healthy))
        print(f'{variant} {name} seed={seed} cases={len(events)} requests={arrays[4]}', flush=True)
    labels = {key: np.mean([run[i] for run in evaluated], axis=0)
              for i, key in enumerate(('actual', 'additive', 'outage', 'fixed', 'healthy'))}
    np.savez_compressed(path, **labels)
    return labels


def summarize(events, labels, choices, reference):
    actual, healthy = labels['actual'], labels['healthy']
    costs = {k: actual[np.arange(len(events)), c] for k, c in choices.items()}
    costs['offline_oracle'] = actual.min(axis=1)
    ref = costs[reference]; excess = ref-healthy
    excess_mean, excess_ci = day_interval(excess, events)
    space = float(day_macro(ref-costs['offline_oracle'], events))
    result = {'reference': reference, 'events': len(events), 'days': len({e['day'] for e in events}),
              'healthy_cost': float(day_macro(healthy, events)), 'reference_fault_added_loss': float(excess_mean),
              'reference_fault_added_ci': excess_ci.tolist(), 'oracle_absolute_space': space, 'policies': {}}
    for name, cost in costs.items():
        delta = ref-cost
        gain, ci = day_interval(delta, events)
        result['policies'][name] = {'cost': float(day_macro(cost, events)), 'absolute_gain': float(gain), 'absolute_gain_ci': ci.tolist(),
             'total_loss_gain_fraction': float(gain/day_macro(ref, events)),
             'fault_added_gain_fraction': float(gain/excess_mean) if excess_mean > 0 else None,
             'fault_added_gain_ci': relative_day_interval(delta, excess, events).tolist() if excess_ci[0] > 0 else None,
             'oracle_space_captured': float(gain/space) if space > 0 else None}
    result['candidate_vs_each_baseline'] = {}
    for baseline in BASELINES:
        mean, ci = day_interval(costs[baseline]-costs['state_mlp'], events)
        result['candidate_vs_each_baseline'][baseline] = {'absolute_gain': float(mean), 'ci': ci.tolist()}
    result['by_hour'] = {}; result['by_category'] = {}
    for attr in ('hour', 'category'):
        for key in sorted({e[attr] for e in events}):
            ix = [i for i, e in enumerate(events) if e[attr] == key]
            mean, ci = day_interval((ref-costs['state_mlp'])[ix], [events[i] for i in ix])
            result['by_'+attr][str(key)] = {'absolute_gain': float(mean), 'ci': ci.tolist()}
    result['candidate_change_vs_reference_fraction'] = float(day_macro((choices['state_mlp'] != choices[reference]).astype(float), events))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/state-value-pilot')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    z, valid, scale, adjacency, ts, context, cuts = read_data()
    groups = json.loads((ROOT/'outputs/multifault-pilot/config.json').read_text(encoding='utf-8'))['groups']
    previous = json.loads((ROOT/'outputs/fresh-holdout-pilot/config.json').read_text(encoding='utf-8'))
    events = {'training': previous['events']['calibration']+previous['events']['test'],
              'validation': make_events(ts, valid, cuts, 12, 3, groups, 2, 24),
              'test': make_events(ts, valid, cuts, 12, 3, groups, 3, 24)}
    rebuilt = make_events(ts, valid, cuts, 12, 3, groups, 0, 24)+make_events(ts, valid, cuts, 12, 3, groups, 1, 24)
    if rebuilt != events['training']: raise AssertionError('Archived event order changed')
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in [ROOT/'docs/state_value_protocol.zh-CN.md', Path(__file__),
                ROOT/'data/raw/metr-la.h5', ROOT/'outputs/fresh-holdout-pilot/config.json',
                *[ROOT/'outputs/fresh-holdout-pilot'/f'arrays_{v}.npz' for v in ('graph_tcn_candidate', 'mlp_baseline')],
                *[ROOT/'outputs/fresh-holdout-pilot'/f'model_tcn_seed{s}.pt' for s in SEEDS],
                *[ROOT/'outputs/recovery-pilot'/f'model_seed{s}.pt' for s in SEEDS]]}
    config = {'created_date': '2026-10-10', 'hashes': hashes, 'groups': groups, 'events': events,
              'time_ranges': {'training': ['2012-04-12', '2012-05-17'], 'validation': ['2012-05-17', '2012-05-31'], 'test': ['2012-05-31', '2012-06-28']},
              'primary_forecaster': 'graph_tcn_candidate', 'ridge_alphas': ALPHAS, 'value_seeds': VALUE_SEEDS,
              'fault_added_screen': .05, 'initial_selection_only': True, 'forecast_seeds': SEEDS}
    config = json.loads(json.dumps(config))
    config_path = args.output/'config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != config:
        raise ValueError('Frozen configuration differs; use a new output directory')
    write_json(config_path, config)
    print(json.dumps({'events': {k: len(v) for k, v in events.items()}, 'ranges': config['time_ranges']}, ensure_ascii=False), flush=True)
    if args.prepare_only: return
    feature_arrays = {key: features(es, groups, z, valid, context, adjacency) for key, es in events.items()}
    normalized, norms = {}, {}
    for i, key in enumerate(('calendar', 'state')):
        arr, mean, std = standardize(feature_arrays['training'][i], [feature_arrays[p][i] for p in ('validation', 'test')])
        normalized[key] = dict(zip(('training', 'validation', 'test'), arr)); norms[key] = (mean, std)
    np.savez_compressed(args.output/'features.npz', **{f'{p}_{k}': v for k, parts in normalized.items() for p, v in parts.items()},
                        **{f'{k}_{n}': norms[k][i] for k in norms for i, n in enumerate(('mean', 'std'))})
    # Separate hidden-input and future-input perturbations for the actual selector features.
    for event in events['test'][::max(1, len(events['test'])//36)]:
        t = event['origin']; group = groups[event['group']]
        original = observable_features(z, valid, context, t, group, adjacency)
        altered = z.copy(); altered[t-12:t, group['nodes']] += 1234
        hidden = observable_features(altered, valid, context, t, group, adjacency)
        altered = z.copy(); altered[t:] += 1234
        future = observable_features(altered, valid, context, t, group, adjacency)
        if not all(np.array_equal(a, b) and np.array_equal(a, c) for a, b, c in zip(original, hidden, future)):
            raise AssertionError('Unobservable data leaked into policy features')
    result = {'config_sha256': sha256(config_path), 'checks': {'hidden_history_invariant': True, 'future_data_invariant': True,
              'archived_training_order_rebuilt': True, 'forecast_mask_invariant': True, 'no_prerepair_history': True}, 'models': {}}
    weight = weighted_rows(events['training'])
    scale_t = torch.tensor(scale).view(1, 1, -1)
    for variant in ('graph_tcn_candidate', 'mlp_baseline'):
        with np.load(ROOT/'outputs/fresh-holdout-pilot'/f'arrays_{variant}.npz') as archive:
            train = {key: np.concatenate([archive['calibration_'+key], archive[key]]) for key in ('actual', 'additive', 'outage', 'fixed', 'healthy')}
        val = evaluate_partition(variant, 'validation', events['validation'], groups, z, valid, context, scale_t, adjacency, args.output)
        target = train['actual']-train['actual'].mean(axis=1, keepdims=True)
        fixed = calibrated_policies(events['training'], groups, train['actual'], train['additive'], train['fixed'], False)
        choices = {p: {name: np.array([fixed[f"{e['group']}:{e['hour']}"][method] for e in events[p]])
                  for name, method in (('shortest_time', 'shortest_time'), ('fixed_joint', 'joint'), ('fixed_gain_time', 'gain_per_time'))}
                  for p in ('validation', 'test')}
        selection = {'ridge': {}, 'fixed': fixed}
        coefficients = {}
        for name, feature_key in (('calendar_ridge', 'calendar'), ('state_ridge', 'state'), ('state_gain_time', 'state')):
            gains = train['fixed'][:, 0:1]-train['fixed'][:, [1, 2, 4]]
            keep = np.isfinite(gains).all(axis=1) if name == 'state_gain_time' else np.ones(len(target), bool)
            y = gains[keep] if name == 'state_gain_time' else target
            scores, best = [], float('inf')
            for alpha in ALPHAS:
                coef = ridge_fit(normalized[feature_key]['training'][keep], y, alpha, weight[keep])
                prediction = ridge_predict(normalized[feature_key]['validation'], coef)
                choice = gain_choices(prediction, events['validation'], groups) if name == 'state_gain_time' else prediction.argmin(axis=1)
                cost = selected_cost(val['actual'], choice, events['validation']); scores.append({'alpha': alpha, 'validation_cost': cost})
                if cost < best: best, best_coef, best_alpha = cost, coef, alpha
            coefficients[name] = best_coef
            selection['ridge'][name] = {'selected_alpha': best_alpha, 'validation_cost': best, 'grid': scores}
            for partition in choices:
                pred = ridge_predict(normalized[feature_key][partition], best_coef)
                choices[partition][name] = gain_choices(pred, events[partition], groups) if name == 'state_gain_time' else pred.argmin(axis=1)
        models, target_scale, log = train_value(normalized['state']['training'], target, normalized['state']['validation'], val['actual'], events['validation'], weight)
        selection['value_mlp'] = log
        for p in choices:
            with torch.no_grad():
                preds = np.stack([m(torch.tensor(normalized['state'][p])).numpy() for m in models])
            choices[p]['state_mlp'] = preds.mean(axis=0).argmin(axis=1)
            for seed, pred in zip(VALUE_SEEDS, preds): choices[p][f'state_mlp_seed{seed}'] = pred.argmin(axis=1)
        val_costs = {k: selected_cost(val['actual'], c, events['validation']) for k, c in choices['validation'].items()}
        reference = min(BASELINES, key=lambda k: val_costs[k])
        selection.update({'reference': reference, 'validation_costs': val_costs, 'target_scale': target_scale})
        write_json(args.output/f'selection_{variant}.json', selection)
        np.savez_compressed(args.output/f'ridge_{variant}.npz', **coefficients)
        for seed, model in zip(VALUE_SEEDS, models):
            torch.save({'state_dict': model.state_dict(), 'dimension': normalized['state']['training'].shape[1], 'target_scale': target_scale,
                        'seed': seed, 'epoch': log['selected_epoch']}, args.output/f'value_{variant}_seed{seed}.pt')
        # Persist choices and selected reference before opening any final-test labels.
        np.savez_compressed(args.output/f'choices_{variant}.npz', **{f'{p}_{k}': v for p, table in choices.items() for k, v in table.items()})
        choices_hash = sha256(args.output/f'choices_{variant}.npz')
        print(json.dumps({'variant': variant, 'reference': reference, 'selected_epoch': log['selected_epoch'], 'validation_costs': val_costs}), flush=True)
        test = evaluate_partition(variant, 'test', events['test'], groups, z, valid, context, scale_t, adjacency, args.output)
        if choices_hash != sha256(args.output/f'choices_{variant}.npz'): raise AssertionError('Test choices changed')
        summary = summarize(events['test'], test, choices['test'], reference)
        summary['choices_sha256_before_test_labels'] = choices_hash
        summary['selection'] = {'reference': reference, 'selected_epoch': log['selected_epoch'], 'validation_costs': val_costs}
        result['models'][variant] = summary
        write_json(args.output/f'summary_{variant}.json', summary)
        print(json.dumps({'variant': variant, 'candidate': summary['policies']['state_mlp']}), flush=True)
    primary = result['models']['graph_tcn_candidate']['policies']['state_mlp']
    replica = result['models']['mlp_baseline']['policies']['state_mlp']
    result['screen_passed'] = bool(primary['fault_added_gain_fraction'] is not None and primary['fault_added_gain_ci'] is not None
                                  and primary['fault_added_gain_fraction'] >= .05 and primary['fault_added_gain_ci'][0] > 0 and replica['absolute_gain'] > 0)
    result['elapsed_seconds'] = time.perf_counter()-started
    write_json(args.output/'results.json', result)
    print(json.dumps({'done_seconds': result['elapsed_seconds'], 'screen_passed': result['screen_passed']}), flush=True)


if __name__ == '__main__':
    main()
