"""Build thesis tables from frozen JSON results, without training or traffic reads."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/pems-bay-thesis-draft'
DOC = ROOT / 'docs/thesis_experiment_discussion.zh-CN.md'
TABLES = ROOT / 'docs/thesis_result_tables.zh-CN.md'
TEMPLATE = ROOT / 'docs/thesis_experiment_discussion.template.zh-CN.md'
MANIFEST = OUT / 'documentation_manifest.json'
METHODS = {
    'dense_speed': '预测速度规则',
    'global_calibration': '全局校准',
    'group_calibration': '可用性分组校准',
    'state_logistic': '状态逻辑回归（Y）',
    'persistent_mlp': '持续低速MLP（Y）',
    'onset_linear': '起点逻辑回归（Z）',
    'onset_mlp': '起点MLP（Z）',
}
CASES = ('natural', 'single_outage', 'group_outage')


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    return digest


def verify_map(mapping):
    for name, digest in mapping.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha(path) != digest:
            raise ValueError(f'Missing or changed evidence: {name}')


def interval(values, multiplier=100, digits=2):
    if values is None:
        return '不可计算'
    return '[' + ', '.join(f'{v * multiplier:.{digits}f}' for v in values) + ']'


def table(headers, rows):
    return '\n'.join([
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
        *['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows],
    ])


def build(full_integrity=False):
    execution = read('outputs/pems-bay-evaluation/execution_freeze.json')
    completed = read('outputs/pems-bay-evaluation/evaluation_results_freeze.json')
    handoff = read('outputs/pems-bay-evaluation/audit_repair_and_handoff_freeze.json')
    evidence_names = [
        'outputs/pems-bay-evaluation/results.json',
        'outputs/pems-bay-evaluation/audit.json',
        'outputs/pems-bay-calibration/selection.json',
        'outputs/pems-bay-forecaster/results.json',
        'outputs/pems-bay-risk/results.json',
        'outputs/pems-bay-evaluation/numeric_audit_comparator_repair.json',
    ]
    expected = {**execution['source_sha256'], **completed['source_sha256'], **handoff['source_sha256']}
    verify_map({name: expected[name] for name in evidence_names})
    if full_integrity:
        verify_map(execution['source_sha256'])
        verify_map(completed['source_sha256'])
        verify_map(handoff['source_sha256'])
    result = read(evidence_names[0])
    audit = read(evidence_names[1])
    selection = read(evidence_names[2])
    forecast = read(evidence_names[3])
    risk = read(evidence_names[4])
    assert audit['passed'] and not result['verdict']['primary_metric_supported']
    assert not result['models_retrained'] and not result['evaluation_threshold_grid_scanned']
    assert result['execution_freeze_sha256'] == sha(ROOT / 'outputs/pems-bay-evaluation/execution_freeze.json')
    assert len(result['evaluation_dates']) == 83

    operation_rows, case_rows, numeric_rows = [], [], []
    for method, label in METHODS.items():
        for grid in ('coarse', 'fine'):
            records = result['evaluation'][method][grid]
            metrics = ('warning_recall_at_least10', 'warning_recall', 'detection_recall', 'false_per_valid_station_day')
            means = {metric: sum(records[case][metric] for case in CASES) / 3 for metric in metrics}
            chosen = selection['methods'][method][grid]['selected']
            setting = f"{(-chosen['threshold'] if method == 'dense_speed' else chosen['threshold']):g}"
            if method == 'dense_speed':
                setting += ' mph'
            setting += f" / {chosen['confirmation']}次"
            exceed = sum(result['calibration_budget_exceedance'][method][grid][case]['exceeded'] for case in CASES)
            operation_rows.append([label, '粗' if grid == 'coarse' else '细', setting,
                *(f'{means[m] * 100:.2f}' for m in metrics[:3]), f'{means[metrics[3]]:.6f}', f'{exceed}/3'])
            numeric_rows.append({'method': method, 'grid': grid, 'mean_scenario_rates': means,
                                 'selected_threshold': chosen['threshold'], 'confirmation': chosen['confirmation'],
                                 'budget_exceeded_scenarios': exceed})
            for case in CASES:
                row = records[case]
                assert row['events'] == 673 and row['detected'] + row['missed_events'] == 673
                assert abs((row['warning_lead_counts']['10'] + row['warning_lead_counts']['15']) / row['events']
                           - row['warning_recall_at_least10']) < 1e-12
                if grid == 'fine':
                    budget = result['calibration_budget_exceedance'][method][grid][case]
                    case_rows.append([label, case, f"{100 * row[metrics[0]]:.2f}",
                        f"{100 * row[metrics[1]]:.2f}", f"{100 * row[metrics[2]]:.2f}",
                        f"{row[metrics[3]]:.6f}", f"{budget['calibration_budget']:.6f}",
                        int(row['missed_events']), int(row['duplicate']), '是' if budget['exceeded'] else '否'])
    operating_headers = ['方法', '网格', '冻结阈值 / 确认', '≥10分钟召回（%）', '全部预警（%）',
                         '含迟报检测（%）', '误报/有效站日', '超预算情景']
    fine_table = table([v for i, v in enumerate(operating_headers) if i != 1],
                       [[v for i, v in enumerate(row) if i != 1] for row in operation_rows if row[1] == '细'])

    primary_rows, service_rows, auxiliary_rows = [], [], []
    for name, item in result['comparisons'].items():
        warning = item['metrics']['warning10_recall']
        label = item['candidate'] + ' − ' + item['baseline']
        if item['primary']:
            primary_rows.append([label, f"{100 * warning['difference']:.2f}", interval(warning['ci975']),
                                 interval(warning['ci95']), '是' if warning['ci975'][0] > 0 else '否'])
            detection = item['metrics']['detection_recall']
            false = item['metrics']['false_rate']
            service_rows.append([label, f"{100 * detection['difference']:.2f}", interval(detection['ci95']),
                                 f"{false['difference']:.6f}", interval(false['ci95'], 1, 6)])
        else:
            auxiliary_rows.append([label, f"{100 * warning['difference']:.2f}", interval(warning['ci95'])])
    forecast_rows = []
    for name, key in [('三种子图TCN集成', 'three_seed_ensemble'), ('最近可见值持久性', 'last_visible_persistence')]:
        row = forecast['validation'][key]
        forecast_rows.append([name, f"{row['mae_mph']:.4f}", *(f'{v:.4f}' for v in row['mae_mph_by_horizon'])])
    risk_rows = []
    for target, values in risk['validation'].items():
        for method, record in values.items():
            risk_rows.append(['Y：持续低速' if target == 'persistent_target_Y' else 'Z：未来起点',
                              METHODS[method], f"{record['bce']:.6f}", f"{record['brier']:.6f}"])
    replacements = {
        'forecast_table': table(['模型', '总体MAE（mph）', '第1步', '第2步', '第3步'], forecast_rows),
        'risk_table': table(['监督目标', '方法', '验证BCE', '验证Brier'], risk_rows),
        'fine_table': fine_table,
        'primary_table': table(['预定主比较', '差（pp）', '97.5%区间（pp）', '描述性95%区间（pp）', '主区间下界>0'], primary_rows),
        'service_table': table(['比较', '检测差（pp）', '95%区间（pp）', '误报差/站日', '95%区间'], service_rows),
        'auxiliary_table': table(['辅助比较', '≥10分钟差（pp）', '描述性95%区间（pp）'], auxiliary_rows),
    }
    chapter = TEMPLATE.read_text(encoding='utf-8')
    for key, content in replacements.items():
        assert chapter.count('{{' + key + '}}') == 1, key
        chapter = chapter.replace('{{' + key + '}}', content)
    assert '{{' not in chapter
    appendix = ('# 论文结果完整附表\n\n'
        '自动生成于已冻结PEMS-BAY v1结果。三情景率等权平均，673个自然代理事件不是2019个独立事件。'
        '原始精度、来源SHA见`outputs/pems-bay-thesis-draft/tables.json`与`documentation_manifest.json`。\n\n'
        '## A. 全部七方法、粗细已选运行点\n\n' + table(operating_headers, operation_rows) +
        '\n\n## B. 细网格三个情景分别报告\n\n' + table(
            ['方法', '情景', '≥10分钟（%）', '全部预警（%）', '检测（%）', '误报/站日', '原预算', '漏检', '重复告警', '超出预算'], case_rows) +
        '\n\n## C. 预定主比较\n\n' + replacements['primary_table'] +
        '\n\n主验收未通过。95%区间不能替代97.5%区间，辅助发现不能升级为主比较。\n\n'
        '## D. 全部辅助长提前量比较\n\n' + replacements['auxiliary_table'] + '\n')
    payload = {'stage': 'documentation_only_from_frozen_results', 'traffic_speed_values_read': False,
        'training_or_threshold_selection_executed': False, 'operating_points': numeric_rows,
        'comparisons': result['comparisons'], 'verdict': result['verdict']}
    outputs = {DOC: chapter, TABLES: appendix,
               OUT / 'tables.json': json.dumps(payload, ensure_ascii=False, indent=2) + '\n'}
    inputs = evidence_names + [
        'outputs/pems-bay-evaluation/execution_freeze.json',
        'outputs/pems-bay-evaluation/evaluation_results_freeze.json',
        'outputs/pems-bay-evaluation/audit_repair_and_handoff_freeze.json',
        TEMPLATE.relative_to(ROOT).as_posix(), Path(__file__).relative_to(ROOT).as_posix(),
    ]
    manifest = {'stage': 'documentation_only_from_frozen_results', 'date': '2026-10-11',
        'source_sha256': {name: sha(ROOT / name) for name in inputs},
        'generated_sha256': {path.relative_to(ROOT).as_posix(): hashlib.sha256(content.encode('utf-8')).hexdigest()
                             for path, content in outputs.items()},
        'evaluation_data_already_analyzed': True, 'primary_metric_supported': False,
        'not_a_new_independent_validation': True}
    return outputs, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Read-only: compare regenerated text and saved manifest')
    parser.add_argument('--full-integrity', action='store_true', help='Also hash every frozen execution dependency, including raw file bytes')
    args = parser.parse_args()
    outputs, manifest = build(args.full_integrity)
    if args.check:
        assert read(MANIFEST.relative_to(ROOT).as_posix()) == manifest, 'Documentation manifest changed'
        for path, content in outputs.items():
            assert path.read_bytes() == content.encode('utf-8'), f'Generated document changed: {path.name}'
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        DOC.parent.mkdir(parents=True, exist_ok=True)
        for path, content in outputs.items():
            path.write_bytes(content.encode('utf-8'))
        MANIFEST.write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print(json.dumps({'documentation': 'verified' if args.check else 'built',
        'generated_documents': len(outputs), 'operating_points': 14, 'fine_scenario_rows': 21,
        'primary_comparisons': 2, 'auxiliary_comparisons': 9, 'primary_metric_supported': False,
        'full_frozen_dependency_integrity_checked': args.full_integrity,
        'traffic_speed_values_decoded': False, 'models_or_thresholds_changed': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
