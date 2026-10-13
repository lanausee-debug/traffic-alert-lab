"""Post-run correction: converged convex baselines, frozen risk MLP."""
import copy
import json
from pathlib import Path
import shutil
import time
import h5py
import numpy as np
import torch
from torch import nn
from recovery_pilot import ROOT, write_json
from state_value_pilot import read_data
from multifault_pilot import sha256
from congestion_alert_pilot import SCENARIOS
from congestion_risk_pilot import (METHODS, BASELINES, RiskModel, prepare_partition, pool,
                                  full_scores, choose, summary_key, operating, aggregate_ci)

OLD = ROOT/'outputs/congestion-risk-pilot'
OUTPUT = ROOT/'outputs/congestion-risk-solver-audit'
ALPHAS = (.0001, .001, .01)


def fit_linear(x, y, xv, yv, path):
    xt, yt = torch.tensor(x), torch.tensor(y)
    logs, best, selected_state, selected_alpha = [], float('inf'), None, None
    for alpha in ALPHAS:
        model = RiskModel(x.shape[1])
        with torch.no_grad():
            model.layers.weight.zero_()
            prior = np.clip(y.mean(), 1e-5, 1-1e-5)
            model.layers.bias.fill_(float(np.log(prior/(1-prior))))
        optimizer = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=300, max_eval=600,
            tolerance_grad=1e-7, tolerance_change=1e-10, line_search_fn='strong_wolfe')
        calls = [0]
        def closure():
            calls[0] += 1
            optimizer.zero_grad(set_to_none=True)
            objective = nn.functional.binary_cross_entropy_with_logits(model(xt), yt)+alpha/2*model.layers.weight.square().sum()
            objective.backward()
            return objective
        optimizer.step(closure)
        objective = float(closure().detach())
        gradient = float(max(p.grad.abs().max().item() for p in model.parameters()))
        with torch.no_grad():
            val_loss = float(nn.functional.binary_cross_entropy_with_logits(model(torch.tensor(xv)), torch.tensor(yv)))
        logs.append({'alpha': alpha, 'validation_bce': val_loss, 'train_objective': objective,
                     'max_abs_gradient': gradient, 'closure_calls': calls[0]})
        if val_loss < best:
            best, selected_state, selected_alpha = val_loss, copy.deepcopy(model.state_dict()), alpha
    model.load_state_dict(selected_state); model.eval()
    record = {'solver': 'LBFGS', 'dimension': x.shape[1], 'nonlinear': False,
              'selected_alpha': selected_alpha, 'validation_bce': best, 'logs': logs,
              'training_rows': len(y), 'validation_rows': len(yv)}
    torch.save({'state_dict': selected_state, **record}, path)
    return model, record


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    started = time.perf_counter(); OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT/'config.json').exists():
        raise ValueError('Existing solver audit preserved')
    old_result = json.loads((OLD/'results.json').read_text(encoding='utf-8'))
    config = copy.deepcopy(old_result['config'])
    config.update({'solver_audit_post_run': True, 'baseline_alphas': ALPHAS, 'linear_solver': 'LBFGS'})
    paths = [Path(__file__), ROOT/'docs/congestion_solver_audit_protocol.zh-CN.md', OLD/'normalization.npz', OLD/'model_selection.json']
    paths += [OLD/f'mlp{s}.pt' for s in (202, 203, 204)]
    for p in paths:
        config['sha256'][str(p.relative_to(ROOT))] = sha256(p)
    write_json(OUTPUT/'config.json', config)
    for filename in ('normalization.npz', 'features.json', 'predictions_training.npz', 'predictions_validation.npz', 'mlp202.pt', 'mlp203.pt', 'mlp204.pt'):
        shutil.copy2(OLD/filename, OUTPUT/filename)
    z, valid, scale, adjacency, ts, context, cuts = read_data()
    original = json.loads((ROOT/'outputs/recovery-pilot/config.json').read_text(encoding='utf-8'))
    mean = np.array(original['normalization_mean'], np.float32)
    with h5py.File(ROOT/'data/raw/metr-la.h5', 'r') as f:
        raw = f['df/block0_values'][:, original['raw_column_indices']].astype(np.float32)
    day = (ts.astype('datetime64[D]')-ts[0].astype('datetime64[D]')).astype(int)
    groups = json.loads((ROOT/'outputs/multifault-pilot/config.json').read_text(encoding='utf-8'))['groups']
    data = raw, z, valid, ts, context, mean, scale, adjacency, day, groups
    at = lambda d: int(np.searchsorted(day, d))
    train, _ = prepare_partition('training', at(42), at(70), 3, data, [], OUTPUT)
    val, _ = prepare_partition('validation', at(70), at(77), 1, data, [], OUTPUT)
    n = np.load(OUTPUT/'normalization.npz'); norm = n['mean'], n['std'], n['single_mean'], n['single_std']
    x = ((pool(train, 'state')-norm[0])/norm[1]).astype(np.float32)
    xv = ((pool(val, 'state')-norm[0])/norm[1]).astype(np.float32)
    single = ((pool(train, 'single')[:, None]-norm[2])/norm[3]).astype(np.float32)
    sv = ((pool(val, 'single')[:, None]-norm[2])/norm[3]).astype(np.float32)
    y, yv = pool(train, 'truth').astype(np.float32), pool(val, 'truth').astype(np.float32)
    gt, gv = pool(train, 'group'), pool(val, 'group')
    logs = {'risk_mlp': old_result['model_selection']['risk_mlp'], 'group_calibration': {}}
    global_model, logs['global_calibration'] = fit_linear(single, y, sv, yv, OUTPUT/'global.pt')
    print('LBFGS global fitted', flush=True)
    group_models = {}
    for g in (0, 1, 2):
        a, b = gt == g, gv == g
        if a.sum() < 100 or len(np.unique(y[a])) < 2 or not b.any():
            logs['group_calibration'][str(g)] = {'fallback': 'global'}; continue
        group_models[str(g)], logs['group_calibration'][str(g)] = fit_linear(single[a], y[a], sv[b], yv[b], OUTPUT/f'group{g}.pt')
    logistic, logs['state_logistic'] = fit_linear(x, y, xv, yv, OUTPUT/'logistic.pt')
    print(f'LBFGS state logistic fitted elapsed={time.perf_counter()-started:.1f}s', flush=True)
    mlps = []
    for seed in (202, 203, 204):
        saved = torch.load(OUTPUT/f'mlp{seed}.pt', weights_only=True)
        model = RiskModel(saved['dimension'], True); model.load_state_dict(saved['state_dict']); model.eval(); mlps.append(model)
    fitted = global_model, group_models, logistic, mlps
    write_json(OUTPUT/'model_selection.json', logs)
    calibration, _ = prepare_partition('calibration', at(77), at(91), 1, data, [], OUTPUT)
    sc = full_scores(calibration, fitted, norm); budgets = old_result['selection']['budgets']
    selections = {m: choose(m, sc[m], calibration, budgets) for m in METHODS}
    def rank(m):
        return (*summary_key(list(selections[m]['selected']['by_scenario'].values())), BASELINES.index(m))
    reference = min(BASELINES, key=rank)
    selection = {'reference': reference, 'candidate': 'risk_mlp', 'budgets': budgets, 'methods': selections}
    write_json(OUTPUT/'selection.json', selection)
    np.savez_compressed(OUTPUT/'scores_calibration.npz', **{f'{m}_{s}': v for m, parts in sc.items() for s, v in parts.items()})
    print(f'calibration reference={reference}; evaluating frozen settings', flush=True)
    evaluation, _ = prepare_partition('evaluation', at(91), at(119), 1, data, [], OUTPUT)
    se = full_scores(evaluation, fitted, norm)
    np.savez_compressed(OUTPUT/'scores_evaluation.npz', **{f'{m}_{s}': v for m, parts in se.items() for s, v in parts.items()})
    summaries, daily_all = {}, {}
    for m in METHODS:
        summaries[m] = {}; op = selections[m]['selected']
        for s in SCENARIOS:
            row, daily = operating(se[m][s], evaluation[s], op['threshold'], op['confirmation'])
            if m != 'dense_speed':
                p = se[m][s]; e = evaluation[s]['eligible']; target = evaluation[s]['truth'][e].astype(np.float64)
                row['brier'] = float(np.mean((p[e]-target)**2))
                row['bce'] = float(np.mean(-target*np.log(np.clip(p[e], 1e-7, 1-1e-7))-(1-target)*np.log(np.clip(1-p[e], 1e-7, 1))))
            summaries[m][s] = row; daily_all[f'{m}_{s}'] = daily
    a = np.stack([daily_all[f'{reference}_{s}'] for s in SCENARIOS]); b = np.stack([daily_all[f'risk_mlp_{s}'] for s in SCENARIOS])
    comparison = aggregate_ci(a, b)
    comparison.update({'mean_warning_recall_difference': float(np.mean([summaries['risk_mlp'][s]['warning_recall']-summaries[reference][s]['warning_recall'] for s in SCENARIOS])),
                       'mean_false_rate_difference': float(np.mean([summaries['risk_mlp'][s]['false_per_valid_station_day']-summaries[reference][s]['false_per_valid_station_day'] for s in SCENARIOS]))})
    result = {'config': config, 'reference': reference, 'candidate': 'risk_mlp', 'selection': selection,
              'evaluation': summaries, 'comparison': comparison, 'model_selection': logs,
              'elapsed_seconds': time.perf_counter()-started, 'new_method_originality_confirmed': False}
    write_json(OUTPUT/'results.json', result); np.savez_compressed(OUTPUT/'daily_metrics.npz', **daily_all)
    selected_gradients = {}
    for method in ('global_calibration', 'state_logistic'):
        selected_gradients[method] = next(r['max_abs_gradient'] for r in logs[method]['logs'] if r['alpha'] == logs[method]['selected_alpha'])
    for group, log in logs['group_calibration'].items():
        if 'selected_alpha' in log:
            selected_gradients[f'group{group}'] = next(r['max_abs_gradient'] for r in log['logs'] if r['alpha'] == log['selected_alpha'])
    audit = {'initial_mlp_weights_unchanged': all(sha256(OLD/f'mlp{s}.pt') == sha256(OUTPUT/f'mlp{s}.pt') for s in (202, 203, 204)),
             'source_hashes_unchanged': all(sha256(ROOT/k) == v for k, v in config['sha256'].items()),
             'selected_linear_max_gradients': selected_gradients,
             'selected_linear_gradient_check_passed': all(v <= 1e-4 for v in selected_gradients.values()),
             'artifact_sha256': {p.name: sha256(p) for p in OUTPUT.iterdir() if p.suffix in ('.npz', '.pt')}, 'audit_status': 'pending recomputation'}
    write_json(OUTPUT/'audit.json', audit)
    print(json.dumps({'seconds': result['elapsed_seconds'], 'reference': reference, 'comparison': comparison, 'gradients': selected_gradients}, indent=2))


if __name__ == '__main__':
    main()
