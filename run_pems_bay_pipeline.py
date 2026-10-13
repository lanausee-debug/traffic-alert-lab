"""Run the original PEMS-BAY pipeline in a new isolated workspace."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SEED = ROOT / 'pipeline_seed'
EXPECTED_SHA = '65d69fb0a2323dba9867179eb7af47c8b814186bc459ff0a4937d21614153c8f'
EXPECTED_BYTES = 135930936
STEPS = (
    ('design', ('freeze_external_validation.py', '--verify')),
    ('binding', ('bind_pems_bay.py',)),
    ('reader_audit', ('audit_pems_bay_reader.py',)),
    ('binding_report', ('summarize_pems_bay_binding.py',)),
    ('forecast_training', ('train_pems_bay_forecaster.py',)),
    ('forecast_audit', ('audit_pems_bay_forecaster.py',)),
    ('forecast_report', ('summarize_pems_bay_forecaster.py',)),
    ('risk_training', ('run_pems_bay_pipeline.py', '--worker', 'risk')),
    ('risk_audit', ('complete_pems_bay_risk.py', '--audit')),
    ('risk_report', ('summarize_pems_bay_risk.py',)),
    ('calibration', ('calibrate_pems_bay_alerts.py',)),
    ('calibration_audit', ('audit_pems_bay_calibration.py',)),
    ('calibration_report', ('summarize_pems_bay_calibration.py',)),
    ('evaluation_freeze', ('freeze_pems_bay_evaluation.py',)),
    ('evaluation_preflight', ('audit_pems_bay_evaluation.py', '--preflight')),
    ('evaluation', ('evaluate_pems_bay_alerts.py',)),
    ('evaluation_audit', ('complete_pems_bay_evaluation_audit.py',)),
    ('evaluation_report', ('summarize_pems_bay_evaluation.py',)),
)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8'))
    os.replace(temporary, path)


def checked_path(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError(f'Path escapes its root: {name}')
    return path


def verify_files(root, entries):
    for name, metadata in entries.items():
        path = checked_path(root, name)
        if not path.is_file() or path.stat().st_size != metadata['bytes'] or sha(path) != metadata['sha256']:
            raise ValueError(f'Missing or changed file: {name}')


def safe_destination(value):
    path = Path(value).resolve()
    area = (ROOT / 'runs').resolve()
    if path == area or not path.is_relative_to(area):
        raise ValueError('Workspace must be a child directory of this project\'s runs/ directory')
    return path


def validate_state(state):
    completed = state['completed_steps']
    order = [name for name, _ in STEPS]
    if completed != order[:len(completed)] or len(completed) > len(order):
        raise ValueError('Pipeline steps are not a valid completed prefix')
    if not state['dataset_already_analyzed'] or state['fresh_independent_validation']:
        raise ValueError('This run must be labeled reproduction of already analyzed data')


def stage_fingerprint(workspace, state):
    manifest = read(workspace / 'pipeline_records/portable_execution_manifest.json')
    verify_files(workspace, manifest['seed_files'])
    if sha(workspace / 'run_pems_bay_pipeline.py') != manifest['runner_sha256']:
        raise ValueError('Portable runner changed after workspace initialization')
    if sha(workspace / 'data/raw/pems-bay.h5') != EXPECTED_SHA:
        raise ValueError('Input bytes changed')
    if state['execution_manifest_sha256'] != sha(workspace / 'pipeline_records/portable_execution_manifest.json'):
        raise ValueError('Portable execution manifest changed')


def initialize(workspace, data):
    if workspace.exists():
        raise ValueError('Workspace already exists; use --resume or another new name')
    seed_manifest = read(SEED / 'seed_manifest.json')
    verify_files(SEED, seed_manifest['files'])
    if data.stat().st_size != EXPECTED_BYTES or sha(data) != EXPECTED_SHA:
        raise ValueError('Input must match the recorded PEMS-BAY HDF5 bytes')
    for name in seed_manifest['files']:
        if Path(name).suffix in {'.pt', '.npz', '.h5'} or (
            name.startswith('outputs/pems-bay-') and name != 'outputs/pems-bay-validation-plan/design_freeze.json'):
            raise ValueError(f'Fitted artifact cannot seed a fresh training workspace: {name}')
    workspace.mkdir(parents=True)
    for name in seed_manifest['files']:
        target = checked_path(workspace, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SEED / name, target)
    # Copying a source script prevents its __file__ based ROOT from pointing at
    # the historic experiment. Inputs may be local, output stages stay isolated.
    if sha(workspace / 'run_pems_bay_pipeline.py') != sha(Path(__file__)):
        raise ValueError('Seed runner differs from the launched runner')
    target_data = workspace / 'data/raw/pems-bay.h5'
    target_data.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(data, target_data)
    if sha(target_data) != EXPECTED_SHA:
        raise ValueError('Data copy failed byte verification')
    records = workspace / 'pipeline_records'
    records.mkdir()
    (records / 'logs').mkdir()
    (workspace / '.tmp').mkdir()
    manifest = {
        'kind': 'engineering_reproduction_before_any_pipeline_speed_decoding', 'created_at_utc': now(),
        'dataset_already_analyzed': True, 'fresh_independent_validation': False,
        'raw_source_path': str(data), 'raw_source_sha256': EXPECTED_SHA,
        'fitted_pems_bay_artifacts_copied': False,
        'original_v1_design_manifest_reused': True,
        'runner_sha256': sha(Path(__file__)), 'seed_manifest_sha256': sha(SEED / 'seed_manifest.json'),
        'seed_files': seed_manifest['files'],
        'predeclared_adapters': {
            'risk_scoring': 'Empty rows yield an empty float32 vector; nonempty inputs call the unchanged score function. Applied before initial training completion.',
            'evaluation_audit': 'Use the recorded original completion comparator: scalar tolerances recurse into lists; integer counts and structures stay exact.',
        },
        'environment': {'python': sys.version, 'executable': sys.executable,
                        'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'h5py', 'matplotlib', 'torch')}},
    }
    write(records / 'portable_execution_manifest.json', manifest)
    state = {'created_at_utc': now(), 'status': 'ready', 'workspace': str(workspace),
        'dataset_already_analyzed': True, 'fresh_independent_validation': False,
        'completed_steps': [], 'step_runs': [],
        'execution_manifest_sha256': sha(records / 'portable_execution_manifest.json')}
    write(records / 'state.json', state)
    return state


def risk_worker():
    from unittest.mock import patch
    import numpy as np
    import train_pems_bay_risk as training
    original = training.score
    def empty_safe(model, x, *args, **kwargs):
        return np.empty(0, dtype=np.float32) if len(x) == 0 else original(model, x, *args, **kwargs)
    with patch.object(training, 'score', empty_safe):
        training.main()
    weights = {p.name: sha(p) for p in sorted(training.OUTPUT.glob('*.pt'))}
    if len(weights) != 11:
        raise AssertionError('The frozen 24-station design should create exactly eleven risk weights')
    compatibility = training.OUTPUT / 'empty_group_adapter.json'
    record = {
        'recorded_at_utc': now(),
        'reason': 'Previously known zero-row availability-group scoring boundary; fix predeclared before this reproduction.',
        'change': 'For zero rows return an empty float32 score; otherwise use the original function.',
        'training_and_model_selection_changed': False, 'evaluation_values_read': False,
        'adapter_source_sha256': sha(ROOT / 'complete_pems_bay_risk.py'),
        'preserved_weight_sha256': weights,
        'applied_before_initial_scoring': True, 'portable_runner_sha256': sha(Path(__file__)),
    }
    if compatibility.exists():
        prior = read(compatibility)
        if prior['preserved_weight_sha256'] != weights or prior['portable_runner_sha256'] != record['portable_runner_sha256']:
            raise ValueError('Existing scoring adapter differs')
    else:
        write(compatibility, record)


def numeric_difference(a, b, location='', failures=None):
    failures = [] if failures is None else failures
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            failures.append({'path': location, 'reason': 'keys differ'})
        else:
            for key in a:
                numeric_difference(a[key], b[key], location + '/' + key, failures)
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            failures.append({'path': location, 'reason': 'list shape differs'})
        else:
            for index, (first, second) in enumerate(zip(a, b)):
                numeric_difference(first, second, location + '/' + str(index), failures)
    elif isinstance(a, float):
        if not isinstance(b, (float, int)) or not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-10):
            failures.append({'path': location, 'reference': a, 'actual': b})
    elif a != b:
        failures.append({'path': location, 'reference': a, 'actual': b})
    return failures


def completed_integrity(workspace):
    for name in ('outputs/pems-bay-evaluation/execution_freeze.json',
                 'outputs/pems-bay-evaluation/evaluation_results_freeze.json'):
        seal = read(workspace / name)
        for source, digest in seal['source_sha256'].items():
            if sha(checked_path(workspace, source)) != digest:
                raise ValueError(f'Frozen stage changed: {source}')
    for directory in ('pems-bay-forecaster', 'pems-bay-risk', 'pems-bay-calibration', 'pems-bay-evaluation'):
        result = read(workspace / 'outputs' / directory / 'results.json')
        for artifact, digest in result['artifact_sha256'].items():
            if sha(checked_path(workspace / 'outputs' / directory, artifact)) != digest:
                raise ValueError(f'Stage artifact changed: {directory}/{artifact}')
    audit = read(workspace / 'outputs/pems-bay-evaluation/audit.json')
    if not audit['passed'] or audit['audited_fixed_scenario_operating_points'] != 42:
        raise ValueError('Complete event audit is not passed')


def comparison(workspace):
    reference = read(workspace / 'reference_metrics.json')
    forecast = read(workspace / 'outputs/pems-bay-forecaster/results.json')
    risk = read(workspace / 'outputs/pems-bay-risk/results.json')
    calibration = read(workspace / 'outputs/pems-bay-calibration/selection.json')
    evaluation = read(workspace / 'outputs/pems-bay-evaluation/results.json')
    actual = {
        'forecast_validation': forecast['validation'],
        'selected_forecast_epochs': [{k: row[k] for k in ('seed', 'selected_epoch')} for row in forecast['model_selection']],
        'risk_validation': risk['validation'], 'calibration_budgets': calibration['budgets'],
        'settings': {method: {grid: {k: row['selected'][k] for k in ('threshold', 'confirmation', 'grid_index')}
                              for grid, row in grids.items()} for method, grids in calibration['methods'].items()},
        'evaluation_dates': evaluation['evaluation_dates'], 'evaluation': evaluation['evaluation'],
        'comparisons': evaluation['comparisons'], 'verdict': evaluation['verdict'],
    }
    selected_reference = {key: reference[key] for key in actual}
    differences = numeric_difference(selected_reference, actual)
    return {'kind': 'engineering_reproduction_numeric_comparison', 'compared_at_utc': now(),
        'matches_original_metrics': not differences, 'difference_count': len(differences),
        'differences': differences, 'absolute_tolerance': 1e-10, 'relative_tolerance': 1e-12,
        'fitted_pems_bay_artifacts_copied': False, 'dataset_already_analyzed': True,
        'fresh_independent_validation': False, 'primary_metric_supported': evaluation['verdict']['primary_metric_supported'],
        'replayed_scenario_operating_points': evaluation['replayed_scenario_operating_points']}


def run_steps(workspace, state):
    records = workspace / 'pipeline_records'
    state_path = records / 'state.json'
    lock = records / 'running.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ValueError('A run lock exists. Check the recorded PID; do not launch another writer into this workspace.')
    with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
        json.dump({'pid': os.getpid(), 'created_at_utc': now()}, handle)
    try:
        for name, arguments in STEPS[len(state['completed_steps']):]:
            stage_fingerprint(workspace, state)
            attempt = 1 + sum(row['step'] == name for row in state['step_runs'])
            log_path = records / 'logs' / f'{name}-{attempt:02}.log'
            row = {'step': name, 'started_at_utc': now(), 'command': [sys.executable, *arguments],
                   'log': log_path.relative_to(workspace).as_posix(), 'status': 'running'}
            state['step_runs'].append(row)
            state['status'] = 'running'
            write(state_path, state)
            print(f'[{len(state["completed_steps"])+1}/{len(STEPS)}] {name} -> {log_path.name}', flush=True)
            started = time.perf_counter()
            env = dict(os.environ)
            env.update({'MPLCONFIGDIR': str(workspace / '.mplconfig'), 'TEMP': str(workspace / '.tmp'),
                        'TMP': str(workspace / '.tmp'), 'TMPDIR': str(workspace / '.tmp'), 'PYTHONUTF8': '1'})
            with log_path.open('xb') as log:
                process = subprocess.Popen([sys.executable, '-u', *arguments], cwd=workspace,
                                           env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    process.wait(timeout=10)
                    row.update({'status': 'interrupted', 'ended_at_utc': now()})
                    state['status'] = 'interrupted'
                    write(state_path, state)
                    raise
            row.update({'ended_at_utc': now(), 'seconds': time.perf_counter()-started,
                        'exit_code': code, 'log_sha256': sha(log_path), 'status': 'passed' if code == 0 else 'failed'})
            if code:
                state['status'] = 'failed'
                write(state_path, state)
                raise RuntimeError(f'{name} failed with exit code {code}; inspect {log_path}')
            stage_fingerprint(workspace, state)
            state['completed_steps'].append(name)
            write(state_path, state)
        completed_integrity(workspace)
        report = comparison(workspace)
        write(records / 'reproduction_comparison.json', report)
        state['status'] = 'completed' if report['matches_original_metrics'] else 'completed_with_numeric_differences'
        state['completed_at_utc'] = now()
        state['comparison_sha256'] = sha(records / 'reproduction_comparison.json')
        write(state_path, state)
        report_text = (
            '# PEMS-BAY从零训练工程复现\n\n'
            '该数据已经分析。本次复现沿用v1设计与预算，在独立目录从零训练；不是新独立确认。\n\n'
            f'- 全部执行阶段：{len(STEPS)}，均成功。\n'
            f'- 数值与原结果相符：{report["matches_original_metrics"]}；差异项：{report["difference_count"]}。\n'
            f'- 原主指标支持：{report["primary_metric_supported"]}。\n'
            '- 未复制PEMS-BAY权重、输入缓存、预测分数或已选运行点作为训练产物。\n'
            '- 零行评分与浮点列表容差为复现开始前已声明的既有修复。旧阶段文件中的首次解封/事后修复措辞描述其局部执行或原研究历史，不表示本次数据重新成为未见数据。\n'
            '- 复现与旧结果匹配不证明算法原创、现场收益或跨平台逐位一致。\n\n'
            '详情见portable_execution_manifest.json、state.json、logs/和reproduction_comparison.json。\n'
        )
        (records / 'report.zh-CN.md').write_text(report_text, encoding='utf-8')
        print(json.dumps({'workspace': str(workspace), 'status': state['status'],
                          'matches_original_metrics': report['matches_original_metrics'],
                          'primary_metric_supported': report['primary_metric_supported']}, ensure_ascii=False), flush=True)
        if not report['matches_original_metrics']:
            raise RuntimeError('Numerical reproduction differs; preserve it without changing models or tolerances')
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, help='New directory below this project\'s runs/')
    parser.add_argument('--data', type=Path, help='Existing PEMS-BAY HDF5 (135930936 bytes, fixed SHA)')
    parser.add_argument('--resume', action='store_true', help='Continue a stopped run with identical source/data')
    parser.add_argument('--verify-only', action='store_true', help='Verify an already completed run without fitting')
    parser.add_argument('--worker', choices=['risk'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not (ROOT / 'pipeline_records/portable_execution_manifest.json').is_file():
            raise ValueError('Worker must run inside an initialized reproduction workspace')
        risk_worker()
        return
    if args.workspace is None:
        parser.error('--workspace is required')
    workspace = safe_destination(args.workspace)
    if args.resume or args.verify_only:
        state = read(workspace / 'pipeline_records/state.json')
        validate_state(state)
        if state['workspace'] != str(workspace):
            raise ValueError('Workspace record differs')
        if sha(workspace / 'run_pems_bay_pipeline.py') != sha(Path(__file__)):
            raise ValueError('Launched runner differs from the recorded workspace runner')
        stage_fingerprint(workspace, state)
        if args.verify_only:
            if state['status'] not in ('completed', 'completed_with_numeric_differences'):
                raise ValueError('Run is not complete')
            completed_integrity(workspace)
            saved = read(workspace / 'pipeline_records/reproduction_comparison.json')
            current = comparison(workspace)
            current['compared_at_utc'] = saved['compared_at_utc']
            if current != saved or sha(workspace / 'pipeline_records/reproduction_comparison.json') != state['comparison_sha256']:
                raise ValueError('Completed comparison changed')
            print(json.dumps({'completed_run_integrity': 'passed', 'fresh_training_executed': False,
                              'matches_original_metrics': saved['matches_original_metrics']}))
            return
        if state['status'] == 'completed':
            raise ValueError('Run already complete; use --verify-only')
    else:
        if args.data is None:
            parser.error('--data is required for a fresh run')
        state = initialize(workspace, args.data.resolve())
    validate_state(state)
    run_steps(workspace, state)


if __name__ == '__main__':
    main()
