"""Exploratory persistent-low-speed alerts using frozen graph TCN weights."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from recovery_pilot import ROOT, write_json
from state_value_pilot import read_data
from fresh_holdout_pilot import MaskedGraphTCN
from multifault_pilot import sha256

SCENARIOS = ('natural', 'single_outage', 'group_outage')
SETTINGS = [(v, k) for v in (20, 25, 30, 35, 40) for k in (1, 2)]
REFERENCE = 'v30_c1'


def name(setting):
    return f'v{setting[0]}_c{setting[1]}'


def runs(bits):
    padded = np.r_[False, bits, False].astype(np.int8)
    return list(zip(np.flatnonzero(np.diff(padded) == 1),
                    np.flatnonzero(np.diff(padded) == -1)))


def alerts(predictions, threshold, confirmation):
    # Natural target validity must never enter this online rule.
    active = predictions.max(axis=1) <= threshold
    if confirmation == 1:
        return active
    result = np.zeros_like(active)
    result[1:] = active[1:] & active[:-1]
    return result


def outage_mask(valid, days, scenario, groups):
    mask = valid.copy()
    fault = np.zeros_like(mask, dtype=bool)
    for d in np.unique(days):
        if scenario == 'natural':
            continue
        nodes = [int(d % valid.shape[1])] if scenario == 'single_outage' else groups[int(d % len(groups))]['nodes']
        rows = np.flatnonzero(days == d)
        fault[np.ix_(rows, nodes)] = True
    mask[fault] = 0
    return mask, fault


def labels(raw, valid, origins, regular):
    future = np.stack([raw[origins+i] for i in range(3)], axis=1)
    eligible = np.stack([valid[origins+i] > 0 for i in range(3)], axis=1).all(axis=1)
    eligible &= regular[:, None]
    truth = (future <= 30).all(axis=1)
    return truth, eligible, future


def infer(models, z, mask, origins, context, mean, scale, batch_size=256):
    predictions = []
    for model in models:
        pieces = []
        with torch.no_grad():
            for at in range(0, len(origins), batch_size):
                ids = origins[at:at+batch_size]
                hist = ids[:, None] + np.arange(-12, 0)[None]
                x = torch.tensor(np.where(mask[hist] > 0, z[hist], 0))
                m = torch.tensor(mask[hist])
                c = torch.tensor(context[ids])
                pieces.append(model(x, m, c).numpy()*scale[None, None]+mean[None, None])
        predictions.append(np.concatenate(pieces))
    return np.stack(predictions).astype(np.float32)


def event_stats(truth, eligible, alarm, day_ids):
    """Complete proxy events, one-to-one start matching, explicit censoring."""
    n_days = int(day_ids.max())+1
    columns = ['exposure', 'events', 'warned', 'detected', 'false', 'duplicate',
               'late', 'censored_events', 'censored_alerts', 'unknown_alerts']
    daily = np.zeros((n_days, len(columns)), dtype=np.float64)
    leads, delays = [], []
    for node in range(truth.shape[1]):
        np.add.at(daily[:, 0], day_ids, eligible[:, node]/288.)
        for lo, hi in runs(eligible[:, node]):
            ts = [(int(lo+a), int(lo+b)) for a, b in runs(truth[lo:hi, node])]
            complete, censored = [], []
            for s, e in ts:
                if s-2 < lo or e == hi:
                    censored.append((s, e))
                    daily[day_ids[s], 7] += 1
                else:
                    complete.append((s, e))
                    daily[day_ids[s], 1] += 1
            used = set()
            for a, b in runs(alarm[lo:hi, node]):
                a, b = int(a+lo), int(b+lo)
                d = day_ids[a]
                if a == lo or b == hi:
                    daily[d, 8] += 1
                    continue
                if any(s-2 <= a < e for s, e in censored):
                    daily[d, 9] += 1
                    continue
                matches = [j for j, (s, e) in enumerate(complete) if s-2 <= a < e]
                unmatched = [j for j in matches if j not in used]
                if unmatched:
                    j = unmatched[0]
                    used.add(j)
                    s, _ = complete[j]
                    event_day = day_ids[s]
                    daily[event_day, 3] += 1
                    if a <= s:
                        daily[event_day, 2] += 1
                        leads.append(5*(s-a+1))
                    else:
                        daily[event_day, 6] += 1
                        delays.append(5*(a-s-1))
                elif matches:
                    daily[d, 5] += 1
                else:
                    daily[d, 4] += 1
    total = daily.sum(axis=0)
    count = int(total[1])
    stats = {col: float(value) for col, value in zip(columns, total)}
    stats.update({'warning_recall': float(total[2]/count) if count else None,
                  'detection_recall': float(total[3]/count) if count else None,
                  'missed_events': int(total[1]-total[3]),
                  'false_per_valid_station_day': float(total[4]/total[0]) if total[0] else None,
                  'warning_lead_mean_minutes': float(np.mean(leads)) if leads else None,
                  'warning_lead_median_minutes': float(np.median(leads)) if leads else None,
                  'late_detection_mean_minutes': float(np.mean(delays)) if delays else None})
    return stats, daily, columns


def point_stats(truth, eligible, alarm):
    tp = int((truth & alarm & eligible).sum())
    fp = int((~truth & alarm & eligible).sum())
    fn = int((truth & ~alarm & eligible).sum())
    return {'valid_points': int(eligible.sum()), 'positive_points': int((truth & eligible).sum()),
            'true_positives': tp, 'false_positives': fp, 'false_negatives': fn,
            'recall': tp/(tp+fn) if tp+fn else None,
            'precision': tp/(tp+fp) if tp+fp else None,
            'false_positive_rate': fp/int((~truth & eligible).sum()) if (~truth & eligible).any() else None}


def evaluate(saved, settings=SETTINGS):
    result, per_day = {}, {}
    truth, eligible, days = saved['truth'], saved['eligible'], saved['day_ids']
    for scenario in SCENARIOS:
        result[scenario], per_day[scenario] = {}, {}
        prediction = saved[f'predictions_{scenario}'].mean(axis=0)
        fault = saved[f'fault_{scenario}']
        for threshold, confirmation in settings:
            key = name((threshold, confirmation))
            alarm = alerts(prediction, threshold, confirmation)
            stats, daily, columns = event_stats(truth, eligible, alarm, days)
            stats['point_all'] = point_stats(truth, eligible, alarm)
            stats['point_fault_nodes'] = point_stats(truth, eligible & fault, alarm)
            stats['point_other_nodes'] = point_stats(truth, eligible & ~fault, alarm)
            result[scenario][key] = stats
            per_day[scenario][key] = daily
    return result, per_day, columns


def select(calibration):
    scores = {}
    for setting in SETTINGS:
        key = name(setting)
        rows = [calibration[s][key] for s in SCENARIOS]
        if any(r['warning_recall'] is None for r in rows):
            raise ValueError('No complete events on calibration; do not select a policy.')
        scores[key] = {'mean_warning_recall': float(np.mean([r['warning_recall'] for r in rows])),
                       'mean_false_per_valid_station_day': float(np.mean([r['false_per_valid_station_day'] for r in rows]))}
    floor = scores[REFERENCE]['mean_warning_recall']
    candidates = [setting for setting in SETTINGS if scores[name(setting)]['mean_warning_recall'] >= floor-1e-12]
    selected = min(candidates, key=lambda setting: (scores[name(setting)]['mean_false_per_valid_station_day'],
                   -scores[name(setting)]['mean_warning_recall'], setting[1], setting[0]))
    return {'selected': name(selected), 'setting': selected, 'reference': REFERENCE,
            'recall_floor': floor, 'calibration_grid': scores}


def paired_ci(reference, candidate, rounds=2000):
    rng = np.random.default_rng(2718)
    draws = rng.integers(0, len(reference), size=(rounds, len(reference)))
    a, b = reference[draws].sum(axis=1), candidate[draws].sum(axis=1)
    false_delta = b[:, 4]/b[:, 0]-a[:, 4]/a[:, 0]
    recall_delta = b[:, 2]/np.maximum(b[:, 1], 1)-a[:, 2]/np.maximum(a[:, 1], 1)
    return {'candidate_minus_reference_false_rate_ci': np.quantile(false_delta, [.025, .975]).tolist(),
            'candidate_minus_reference_warning_recall_ci': np.quantile(recall_delta, [.025, .975]).tolist()}


def semantic_checks():
    pred = np.full((12, 3, 1), 50., np.float32)
    pred[3:6] = 20
    assert np.flatnonzero(alerts(pred, 30, 1)[:, 0]).tolist() == [3, 4, 5]
    assert np.flatnonzero(alerts(pred, 30, 2)[:, 0]).tolist() == [4, 5]
    truth = np.zeros((20, 1), bool); truth[6:10] = True
    valid = np.ones_like(truth)
    alarm = np.zeros_like(truth); alarm[4:6] = True; alarm[8:9] = True; alarm[13:14] = True
    s, _, _ = event_stats(truth, valid, alarm, np.zeros(20, int))
    assert (s['events'], s['warned'], s['false'], s['duplicate']) == (1, 1, 1, 1)
    assert s['warning_lead_mean_minutes'] == 15
    valid[:6] = False
    s, _, _ = event_stats(truth, valid, alarm, np.zeros(20, int))
    assert s['events'] == 0 and s['censored_events'] == 1 and s['unknown_alerts'] == 1
    return {'confirmation_and_matching_examples': True, 'censoring_example': True}


def hidden_check(model, z, mask, context, origins):
    ids = origins[np.linspace(0, len(origins)-1, 8, dtype=int)]
    hist = ids[:, None]+np.arange(-12, 0)[None]
    x, m = torch.tensor(z[hist]), torch.tensor(mask[hist])
    modified = torch.where(m > 0, x, torch.full_like(x, 1e5))
    with torch.no_grad():
        a = model(x, m, torch.tensor(context[ids]))
        b = model(modified, m, torch.tensor(context[ids]))
    if not torch.equal(a, b):
        raise AssertionError('Hidden-value leakage')


def fmt(value, digits=3):
    return '无可评值' if value is None else f'{value:.{digits}f}'


def report(output, result):
    selected = result['selection']['selected']
    lines = ['# 拥堵告警探索实验：第一阶段', '',
             '日期：2026-10-10。当前研究方向已选择；本阶段只检查可行性和简单规则，没有训练条件风险MLP。', '',
             '冻结三模型图TCN、24站、30 mph持续低速代理标签。校准14天，探索评测28天；这些日期曾在维修课题中分析过，本次不是独立确认实验。', '',
             f'校准选定共同规则：`{selected}`；参考：`{REFERENCE}`。速度单位mph，c表示连续确认次数。', '',
             '| 情景 | 完整事件 | 参考预警召回 | 候选预警召回 | 参考误报/有效站日 | 候选误报/有效站日 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for scenario in SCENARIOS:
        a = result['evaluation'][scenario][REFERENCE]
        b = result['evaluation'][scenario][selected]
        lines.append(f"| {scenario} | {int(a['events'])} | {fmt(a['warning_recall'])} | {fmt(b['warning_recall'])} | {fmt(a['false_per_valid_station_day'])} | {fmt(b['false_per_valid_station_day'])} |")
    lines += ['', '## 覆盖、匹配及限制', '',
              '预警召回只计事件发生前5/10/15分钟开启的告警。晚于起点的告警另计迟报；事件内重启告警单列重复数。原始目标未知片段和边界删失明示，未把它们当通畅或无误报。提前量是成功预警事件上的条件统计，必须结合召回阅读。', '']
    for scenario in SCENARIOS:
        b = result['evaluation'][scenario][selected]
        ci = result['paired_intervals'][scenario]
        lines += [f'### {scenario}', '',
                  f"有效站日{fmt(b['exposure'])}；完整事件{int(b['events'])}；事件删失{int(b['censored_events'])}；告警删失{int(b['censored_alerts'])}；不可判定告警{int(b['unknown_alerts'])}。",
                  f"预警{int(b['warned'])}；迟报{int(b['late'])}；未检测{b['missed_events']}；误報{int(b['false'])}；重复告警{int(b['duplicate'])}。成功预警平均提前{fmt(b['warning_lead_mean_minutes'])}分钟。",
                  f"候选−参考误报率差95%区间{ci['candidate_minus_reference_false_rate_ci']}；预警召回差区间{ci['candidate_minus_reference_warning_recall_ci']}。", '']
    lines += ['## 对选题的含义', '',
              '上述是传统阈值与持续确认的成绩，不能当成论文新方法。如果误报下降而召回或提前量变差，也不能称为同等服务下的改进。条件风险方法尚未实现，其创新及有效性仍待验证。', '',
              '下一阶段应从固定的强简单策略出发，比较全局校准、缺测分组校准、逻辑回归与小MLP；选参仍只用校准段。后续最终方法需要新的独立数据，旧评测不能继续充当确认性测试。', '',
              '原始速度产生的是持续低速代理标签，人工故障是预定模拟，不是现场事故、设备故障或运营数据。本阶段没有交通控制、车辆延误或经济收益验证。', '',
              '结果：results.json；固定选择：selection.json；校准/评测预测：predictions_*.npz；覆盖及阈值敏感性：availability.json；日级统计：daily_metrics.npz；审核：audit.json。']
    (output/'report.zh-CN.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for label, key in [('Fixed 30 mph', REFERENCE), ('Calibration selected', selected)]:
        x = np.arange(3)+(-.18 if label.startswith('Fixed') else .18)
        axes[0].bar(x, [result['evaluation'][s][key]['warning_recall'] for s in SCENARIOS], .35, label=label)
        axes[1].bar(x, [result['evaluation'][s][key]['false_per_valid_station_day'] for s in SCENARIOS], .35, label=label)
    for ax in axes:
        ax.set_xticks(np.arange(3), ['Natural', 'Single outage', 'Group outage'])
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    axes[0].set_ylabel('Event warning recall'); axes[0].set_ylim(0, 1)
    axes[1].set_ylabel('False alert episodes / valid station-day')
    axes[0].legend(fontsize=8)
    fig.suptitle('Exploratory congestion alerts - previously analysed METR-LA dates')
    fig.tight_layout()
    fig.savefig(output/'summary.png', dpi=170)
    fig.savefig(output/'summary.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/congestion-alert-pilot-v2')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--recompute', action='store_true')
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    z, valid, scale, adjacency, timestamps, context, cuts = read_data()
    original = json.loads((ROOT/'outputs/recovery-pilot/config.json').read_text(encoding='utf-8'))
    mean = np.array(original['normalization_mean'], np.float32)
    with h5py.File(ROOT/'data/raw/metr-la.h5', 'r') as h5:
        raw = h5['df/block0_values'][:, original['raw_column_indices']].astype(np.float32)
    groups = json.loads((ROOT/'outputs/multifault-pilot/config.json').read_text(encoding='utf-8'))['groups']
    paths = [ROOT/'docs/congestion_pilot_protocol.zh-CN.md', Path(__file__), ROOT/'data/raw/metr-la.h5',
             ROOT/'state_value_pilot.py', ROOT/'fresh_holdout_pilot.py', ROOT/'recovery_pilot.py',
             ROOT/'outputs/recovery-pilot/config.json', ROOT/'outputs/multifault-pilot/config.json']
    paths += [ROOT/'outputs/fresh-holdout-pilot'/f'model_tcn_seed{seed}.pt' for seed in (42, 43, 44)]
    config = {'date': '2026-10-10', 'exploratory': True, 'prior_all_dates_used': True,
              'title': '面向观测缺失的交通拥堵告警风险校准与事件级决策方法研究',
              'stage': 'event feasibility and simple thresholds; no new risk model',
              'raw_columns': original['raw_column_indices'], 'seeds': [42, 43, 44],
              'settings': SETTINGS, 'scenarios': SCENARIOS, 'proxy_speed_threshold_mph': 30,
              'history_steps': 12, 'future_steps': 3, 'partition_days': {'calibration': [77, 91], 'evaluation': [91, 119]},
              'sha256': {str(p.relative_to(ROOT)): sha256(p) for p in paths}}
    # Tuples serialize as lists; normalize before frozen equality comparison.
    config = json.loads(json.dumps(config, ensure_ascii=False))
    if (output/'config.json').exists():
        if json.loads((output/'config.json').read_text(encoding='utf-8')) != config:
            raise ValueError('Frozen config changed; use a new output directory.')
    else:
        write_json(output/'config.json', config)
    if args.prepare_only:
        print(json.dumps(config, ensure_ascii=False, indent=2)); return
    if (output/'results.json').exists() and not args.recompute:
        raise ValueError('Existing run preserved. Use --recompute or a new output directory.')
    audit = semantic_checks()
    day_absolute = (timestamps.astype('datetime64[D]')-timestamps[0].astype('datetime64[D]')).astype(int)
    models = []
    for seed in (42, 43, 44):
        model = MaskedGraphTCN(12, 3, adjacency)
        model.load_state_dict(torch.load(ROOT/'outputs/fresh-holdout-pilot'/f'model_tcn_seed{seed}.pt', weights_only=True)['state_dict'])
        model.eval(); models.append(model)
    sections, daily_all, availability = {}, {}, {}
    for partition, start, end in [('calibration', cuts[2], cuts[3]), ('evaluation', cuts[3], cuts[4])]:
        cache = output/f'predictions_{partition}.npz'
        if args.recompute:
            with np.load(cache) as data:
                saved = {k: data[k] for k in data.files}
        else:
            origins = np.arange(start+12, end-2)
            hist = origins[:, None]+np.arange(-12, 3)[None]
            regular = (np.diff(timestamps[hist], axis=1) == np.timedelta64(5, 'm')).all(axis=1)
            truth, eligible, future = labels(raw, valid, origins, regular)
            dates = np.unique(timestamps[origins].astype('datetime64[D]'))
            day_ids = np.searchsorted(dates, timestamps[origins].astype('datetime64[D]'))
            saved = {'origins': origins, 'truth': truth, 'eligible': eligible, 'day_ids': day_ids,
                     'dates': dates.astype(str), 'future_raw': future}
            for scenario in SCENARIOS:
                mask, fault = outage_mask(valid, day_absolute, scenario, groups)
                predictions = infer(models, z, mask, origins, context, mean, scale)
                saved[f'predictions_{scenario}'] = predictions
                saved[f'fault_{scenario}'] = fault[origins-1]
                print(f'{partition} {scenario}: origins={len(origins)} elapsed={time.perf_counter()-started:.1f}s', flush=True)
            np.savez_compressed(cache, **saved)
        # Verify cached targets against raw data; eligibility is common to scenarios.
        origins = saved['origins']
        hist = origins[:, None]+np.arange(-12, 3)[None]
        regular = (np.diff(timestamps[hist], axis=1) == np.timedelta64(5, 'm')).all(axis=1)
        truth, eligible, future = labels(raw, valid, origins, regular)
        assert np.array_equal(truth, saved['truth']) and np.array_equal(eligible, saved['eligible'])
        assert (hist[:, :12].max(axis=1) < origins).all()
        for scenario in SCENARIOS:
            mask, fault = outage_mask(valid, day_absolute, scenario, groups)
            assert np.array_equal(fault[origins-1], saved[f'fault_{scenario}'])
            for model in models:
                hidden_check(model, z, mask, context, origins)
        sensitivity = {}
        for threshold in (20, 30, 40):
            proxy = (future <= threshold).all(axis=1)
            stats, _, _ = event_stats(proxy, eligible, np.zeros_like(proxy), saved['day_ids'])
            sensitivity[str(threshold)] = {'positive_points': int((proxy & eligible).sum()),
                'complete_events': int(stats['events']), 'censored_events': int(stats['censored_events'])}
        sections[partition], daily, columns = evaluate(saved)
        availability[partition] = {'origins': len(origins), 'days': len(saved['dates']),
            'eligible_fraction': float(eligible.mean()), 'valid_station_days': float(eligible.sum()/288),
            'coverage_by_day': {str(d): float(eligible[saved['day_ids'] == j].mean()) for j, d in enumerate(saved['dates'])},
            'descriptive_threshold_sensitivity': sensitivity}
        for scenario in SCENARIOS:
            for key, array in daily[scenario].items():
                daily_all[f'{partition}_{scenario}_{key}'] = array
        if partition == 'calibration':
            selection = json.loads(json.dumps(select(sections[partition])))
            if args.recompute:
                assert selection == json.loads((output/'selection.json').read_text(encoding='utf-8'))
            else:
                write_json(output/'selection.json', selection)
            print(json.dumps({'selection': selection['selected'], 'calibration_recall_floor': selection['recall_floor']}, ensure_ascii=False), flush=True)
    selected = selection['selected']
    intervals = {s: paired_ci(daily_all[f'evaluation_{s}_{REFERENCE}'], daily_all[f'evaluation_{s}_{selected}']) for s in SCENARIOS}
    audit.update({'hidden_mask_values_invariant': True, 'strict_past_history': True,
                  'targets_and_coverage_recomputed': True, 'no_new_forecaster_training': True,
                  'selection_calibration_only': True, 'source_data_model_hashes_unchanged': all(sha256(ROOT/k) == v for k, v in config['sha256'].items()),
                  'prediction_cache_sha256': {p.name: sha256(p) for p in output.glob('predictions_*.npz')}})
    assert audit['source_data_model_hashes_unchanged']
    result = {'config': config, 'selection': selection, **sections, 'paired_intervals': intervals,
              'daily_columns': columns, 'availability': availability,
              'elapsed_seconds': time.perf_counter()-started, 'new_method_validated': False}
    write_json(output/'availability.json', availability)
    write_json(output/'results.json', result)
    write_json(output/'audit.json', audit)
    np.savez_compressed(output/'daily_metrics.npz', **daily_all)
    report(output, result)
    print(json.dumps({'done_seconds': result['elapsed_seconds'], 'selected': selected,
        'evaluation': {s: {key: sections['evaluation'][s][selected][key] for key in ('events', 'warning_recall', 'false_per_valid_station_day')} for s in SCENARIOS}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
