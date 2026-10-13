"""Controlled risk-head feature subsets; frozen traffic predictions and labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import time
import h5py
import numpy as np
import torch

from recovery_pilot import ROOT, write_json
from state_value_pilot import read_data
from multifault_pilot import sha256
from congestion_alert_pilot import SCENARIOS
from congestion_risk_pilot import (BASE, RiskModel, prepare_partition, pool, fit, score, sigmoid,
                                  choose, operating, hidden_features_check)
from congestion_solver_audit import fit_linear

OLD = ROOT/'outputs/congestion-risk-solver-audit'
OUTPUT = ROOT/'outputs/congestion-risk-ablation-v2'
SEEDS = (202, 203, 204)
P, C, N, V, A = list(range(8)), list(range(20, 48)), list(range(15, 19)), [8, 9, 10, 11, 14], [12, 13, 19, 48, 49, 50]
SUBSETS = {
    'context_only': C,
    'forecast_only': P,
    'forecast_context': P+C,
    'forecast_context_neighbor': P+C+N,
    'forecast_context_neighbor_availability': P+C+N+A,
    'full_without_availability': P+C+N+V,
    'full_without_context': P+N+V+A,
    'full': P+C+N+V+A,
}
SUBSETS = {key: sorted(columns) for key, columns in SUBSETS.items()}
CONTRASTS = [
    ('full', 'full_without_availability', True),
    ('full', 'full_without_context', True),
    ('forecast_context', 'forecast_only', False),
    ('forecast_context_neighbor', 'forecast_context', False),
    ('forecast_context_neighbor_availability', 'forecast_context_neighbor', False),
    ('full_without_availability', 'forecast_context_neighbor', False),
    ('full', 'forecast_context_neighbor_availability', False),
]


def load_data():
    z, valid, scale, adjacency, ts, context, cuts = read_data()
    cfg = json.loads((ROOT/'outputs/recovery-pilot/config.json').read_text(encoding='utf-8'))
    mean = np.array(cfg['normalization_mean'], np.float32)
    with h5py.File(ROOT/'data/raw/metr-la.h5', 'r') as f:
        raw = f['df/block0_values'][:, cfg['raw_column_indices']].astype(np.float32)
    groups = json.loads((ROOT/'outputs/multifault-pilot/config.json').read_text(encoding='utf-8'))['groups']
    day = (ts.astype('datetime64[D]')-ts[0].astype('datetime64[D]')).astype(int)
    return raw, z, valid, ts, context, mean, scale, adjacency, day, groups


def load_parts(data):
    parts = {}
    day = data[8]
    for name, a, b, stride in [('training', 42, 70, 3), ('validation', 70, 77, 1),
                                ('calibration', 77, 91, 1), ('evaluation', 91, 119, 1)]:
        start, end = int(np.searchsorted(day, a)), int(np.searchsorted(day, b))
        parts[name], names = prepare_partition(name, start, end, stride, data, [], OLD)
    return parts, names


def load_model(path):
    saved = torch.load(path, weights_only=True)
    model = RiskModel(saved['dimension'], saved['nonlinear'])
    model.load_state_dict(saved['state_dict']); model.eval()
    return model, saved


def model_scores(parts, fitted, mean, std):
    scores = {}
    for variant, columns in SUBSETS.items():
        for family in ('linear', 'mlp'):
            key = f'{family}_{variant}'
            scores[key] = {}
            for scenario in SCENARIOS:
                p = parts[scenario]
                values = p['state'].reshape(-1, 51) if variant == 'full' else p['state'][:, :, columns].reshape(-1, len(columns))
                x = ((values-mean[columns])/std[columns]).astype(np.float32)
                models = fitted[key]
                probability = np.mean([sigmoid(score(model, x)) for model in models], axis=0)
                scores[key][scenario] = probability.reshape(p['truth'].shape)
    return scores


def evaluate(scores, parts, selections):
    summaries, daily_all = {}, {}
    for key, scenarios in scores.items():
        summaries[key] = {}
        op = selections[key]['selected']
        for scenario, probability in scenarios.items():
            row, daily = operating(probability, parts[scenario], op['threshold'], op['confirmation'])
            e = parts[scenario]['eligible']; y = parts[scenario]['truth'][e].astype(np.float64)
            p = probability[e]
            row['brier'] = float(np.mean((p-y)**2))
            row['bce'] = float(np.mean(-y*np.log(np.clip(p, 1e-7, 1-1e-7))-(1-y)*np.log(np.clip(1-p, 1e-7, 1))))
            summaries[key][scenario] = row; daily_all[f'{key}_{scenario}'] = daily
    return summaries, daily_all


def contrast(a, b, primary):
    """a minus b, common date draws across scenarios and policy outputs."""
    rng = np.random.default_rng(8120)
    draws = rng.integers(0, a.shape[1], size=(2000, a.shape[1]))
    ar, br = a[:, draws].sum(axis=2), b[:, draws].sum(axis=2)
    at, bt = a.sum(axis=1), b.sum(axis=1)
    out = {'primary': primary}
    for label, numerator, denominator in [('warning_recall', 2, 1), ('detection_recall', 3, 1), ('false_rate', 4, 0)]:
        differences = (ar[:, :, numerator]/np.maximum(ar[:, :, denominator], 1)-br[:, :, numerator]/np.maximum(br[:, :, denominator], 1)).mean(axis=0)
        point = (at[:, numerator]/np.maximum(at[:, denominator], 1)-bt[:, numerator]/np.maximum(bt[:, denominator], 1)).mean()
        out[label+'_difference'] = float(point)
        out[label+'_ci95'] = np.quantile(differences, [.025, .975]).tolist()
        if primary:
            out[label+'_ci975'] = np.quantile(differences, [.0125, .9875]).tolist()
    return out


def contrasts(daily):
    results = {}
    for family in ('mlp', 'linear'):
        for larger, smaller, primary in CONTRASTS:
            a = np.stack([daily[f'{family}_{larger}_{s}'] for s in SCENARIOS])
            b = np.stack([daily[f'{family}_{smaller}_{s}'] for s in SCENARIOS])
            results[f'{family}:{larger}-{smaller}'] = contrast(a, b, primary and family == 'mlp')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume-from', type=Path, help='Reuse completed training checkpoints after a scoring implementation repair')
    args = parser.parse_args(); output = args.output
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    paths = [Path(__file__), ROOT/'docs/congestion_ablation_protocol.zh-CN.md', ROOT/'congestion_risk_pilot.py',
             ROOT/'congestion_solver_audit.py', ROOT/'congestion_alert_pilot.py', ROOT/'state_value_pilot.py',
             ROOT/'fresh_holdout_pilot.py', ROOT/'recovery_pilot.py', ROOT/'data/raw/metr-la.h5',
             ROOT/'outputs/recovery-pilot/config.json', ROOT/'outputs/multifault-pilot/config.json',
             OLD/'normalization.npz', OLD/'features.json', OLD/'config.json', OLD/'results.json', OLD/'logistic.pt',
             OLD/'predictions_training.npz', OLD/'predictions_validation.npz',
             BASE/'predictions_calibration.npz', BASE/'predictions_evaluation.npz']
    paths += [OLD/f'mlp{s}.pt' for s in SEEDS]
    paths += [ROOT/'outputs/fresh-holdout-pilot'/f'model_tcn_seed{s}.pt' for s in (42, 43, 44)]
    resume = args.resume_from.resolve() if args.resume_from else None
    if resume:
        paths.extend(sorted(resume.glob('*.pt')))
        paths.append(resume/'model_selection.json')
    config = {'date': '2026-10-10', 'exploratory': True, 'all_dates_previously_used': True,
              'subsets': SUBSETS, 'contrasts': CONTRASTS, 'seeds': SEEDS, 'frozen_full_reused': True,
              'resumed_training_checkpoints': str(resume) if resume else None,
              'sha256': {str(p.relative_to(ROOT)): sha256(p) for p in paths}}
    config = json.loads(json.dumps(config))
    if (output/'config.json').exists():
        assert config == json.loads((output/'config.json').read_text(encoding='utf-8')), 'Use a new output directory'
    else:
        write_json(output/'config.json', config)
    if args.prepare_only:
        print(json.dumps({'subsets': {k: len(v) for k, v in SUBSETS.items()}, 'config': str(output/'config.json')})); return
    if (output/'results.json').exists():
        raise ValueError('Existing run preserved')
    data = load_data(); parts, names = load_parts(data)
    for partition in ('training', 'validation'):
        hidden_features_check(parts[partition], data)
    n = np.load(OLD/'normalization.npz'); mean, std = n['mean'], n['std']
    xt, xv = pool(parts['training'], 'state'), pool(parts['validation'], 'state')
    calculated_std = xt.std(axis=0); calculated_std[calculated_std < 1e-5] = 1
    assert np.array_equal(xt.mean(axis=0), mean) and np.array_equal(calculated_std, std)
    y, yv = pool(parts['training'], 'truth').astype(np.float32), pool(parts['validation'], 'truth').astype(np.float32)
    np.savez_compressed(output/'normalization.npz', mean=mean, std=std)
    write_json(output/'features.json', {k: [names[i] for i in columns] for k, columns in SUBSETS.items()})
    fitted, logs, gradients = {}, {}, {}
    old_logs = json.loads((OLD/'model_selection.json').read_text(encoding='utf-8'))
    resume_logs = json.loads((resume/'model_selection.json').read_text(encoding='utf-8')) if resume else {}
    for variant, columns in SUBSETS.items():
        key = f'linear_{variant}'
        if variant == 'full':
            shutil.copy2(OLD/'logistic.pt', output/f'{key}.pt')
            model, saved = load_model(output/f'{key}.pt'); record = old_logs['state_logistic']
        else:
            x = ((xt[:, columns]-mean[columns])/std[columns]).astype(np.float32)
            v = ((xv[:, columns]-mean[columns])/std[columns]).astype(np.float32)
            if resume:
                shutil.copy2(resume/f'{key}.pt', output/f'{key}.pt')
                model, saved = load_model(output/f'{key}.pt'); record = resume_logs[key]
            else:
                model, record = fit_linear(x, y, v, yv, output/f'{key}.pt')
        fitted[key], logs[key] = [model], record
        gradients[key] = next(r['max_abs_gradient'] for r in record['logs'] if r['alpha'] == record['selected_alpha'])
        key = f'mlp_{variant}'; fitted[key], logs[key] = [], []
        for seed in SEEDS:
            path = output/f'{key}_{seed}.pt'
            if variant == 'full':
                shutil.copy2(OLD/f'mlp{seed}.pt', path)
                model, saved = load_model(path); record = next(r for r in old_logs['risk_mlp'] if r['seed'] == seed)
            else:
                if resume:
                    shutil.copy2(resume/path.name, path)
                    model, saved = load_model(path); record = next(r for r in resume_logs[key] if r['seed'] == seed)
                else:
                    model, record = fit(x, y, v, yv, True, seed, path)
            fitted[key].append(model); logs[key].append(record)
        print(f'{variant}: {len(columns)} features, models selected, elapsed={time.perf_counter()-started:.1f}s', flush=True)
    write_json(output/'model_selection.json', logs)
    scores_cal = model_scores(parts['calibration'], fitted, mean, std)
    with np.load(OLD/'scores_calibration.npz') as cached:
        for family, method in (('mlp', 'risk_mlp'), ('linear', 'state_logistic')):
            for scenario in SCENARIOS:
                assert np.array_equal(scores_cal[f'{family}_full'][scenario], cached[f'{method}_{scenario}'])
    old_results = json.loads((OLD/'results.json').read_text(encoding='utf-8'))
    budgets = old_results['selection']['budgets']
    selections = {key: choose(key, scenarios, parts['calibration'], budgets) for key, scenarios in scores_cal.items()}
    write_json(output/'selection.json', {'budgets': budgets, 'methods': selections})
    np.savez_compressed(output/'scores_calibration.npz', **{f'{k}_{s}': v for k, rows in scores_cal.items() for s, v in rows.items()})
    print('all calibration choices saved; evaluating frozen settings', flush=True)
    scores_eval = model_scores(parts['evaluation'], fitted, mean, std)
    with np.load(OLD/'scores_evaluation.npz') as cached:
        for family, method in (('mlp', 'risk_mlp'), ('linear', 'state_logistic')):
            for scenario in SCENARIOS:
                assert np.array_equal(scores_eval[f'{family}_full'][scenario], cached[f'{method}_{scenario}'])
    np.savez_compressed(output/'scores_evaluation.npz', **{f'{k}_{s}': v for k, rows in scores_eval.items() for s, v in rows.items()})
    summaries, daily = evaluate(scores_eval, parts['evaluation'], selections)
    differences = contrasts(daily)
    result = {'config': config, 'selection': {'budgets': budgets, 'methods': selections}, 'evaluation': summaries,
              'contrasts': differences, 'elapsed_seconds': time.perf_counter()-started,
              'new_method_originality_confirmed': False}
    write_json(output/'results.json', result); np.savez_compressed(output/'daily_metrics.npz', **daily)
    audit = {'source_hashes_unchanged': all(sha256(ROOT/k) == v for k, v in config['sha256'].items()),
             'normalization_training_only_checked': True, 'hidden_and_future_features_invariant': True,
             'full_probabilities_equal_old': True, 'linear_selected_gradients': gradients,
             'linear_gradient_check_passed': all(v <= 1e-4 for v in gradients.values()),
             'artifact_sha256': {p.name: sha256(p) for p in output.iterdir() if p.suffix in ('.pt', '.npz')},
             'audit_status': 'independent recomputation pending'}
    assert audit['source_hashes_unchanged']
    write_json(output/'audit.json', audit)
    print(json.dumps({'seconds': result['elapsed_seconds'], 'primary_contrasts': {k: v for k, v in differences.items() if v['primary']}, 'gradients_ok': audit['linear_gradient_check_passed']}, indent=2))


if __name__ == '__main__':
    main()
