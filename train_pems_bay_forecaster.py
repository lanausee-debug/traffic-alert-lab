"""Train the three frozen-budget PEMS-BAY TCNs; evaluation remains sealed."""
from __future__ import annotations
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from unittest.mock import patch
import h5py
import numpy as np
import torch
from freeze_external_validation import sha256
from verify_pems_bay_binding import main as verify_binding
from pems_bay_reader import SealedPartitionReader
from pems_bay_windows import forecast_dataset
from fresh_holdout_pilot import MaskedGraphTCN, train_tcn
import fresh_holdout_pilot
from recovery_pilot import artificial_mask, latest

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT/'outputs/pems-bay-forecaster'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


@contextlib.contextmanager
def guard_forecast_reads(reader, interceptions):
    original = h5py.Dataset.__getitem__
    allowed = {reader.bounds[name]: name for name in ('forecast_train', 'forecast_validation')}
    columns = sorted(reader.columns.tolist())
    def guarded(dataset, key, *args, **kwargs):
        if dataset.name == '/speed/block0_values':
            if not isinstance(key, tuple) or not isinstance(key[0], slice):
                raise PermissionError('Training cannot read an unbounded speed array.')
            rows = key[0]
            bounds = (rows.start, rows.stop)
            if bounds not in allowed or rows.step not in (None, 1) or np.asarray(key[1]).tolist() != columns:
                raise PermissionError('Training speed request is outside the two bound forecast partitions.')
            interceptions.append({'partition': allowed[bounds], 'row_start': rows.start,
                                  'row_stop_exclusive': rows.stop, 'columns': columns})
        return original(dataset, key, *args, **kwargs)
    with patch.object(h5py.Dataset, '__getitem__', guarded):
        yield


def predict(model, data, mask):
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(data), 128):
            x, _, _, _, context = [value[start:start+128] for value in data.tensors]
            pieces.append(model(x, mask[start:start+128], context).numpy())
    return np.concatenate(pieces)


def error_summary(prediction, target, valid, scale):
    error = np.abs(prediction-target)*scale[None, None, :]*valid
    counts = valid.sum(axis=(0, 2), dtype=np.float64)
    totals = error.sum(axis=(0, 2), dtype=np.float64)
    if np.any(counts <= 0):
        raise ValueError('Prediction validation has no valid targets.')
    return {'mae_mph': float(totals.sum()/counts.sum()),
            'mae_mph_by_horizon': (totals/counts).tolist(), 'valid_targets': int(counts.sum())}


def load_model(seed, adjacency, path):
    model = MaskedGraphTCN(12, 3, adjacency)
    model.load_state_dict(torch.load(path, weights_only=True)['state_dict'])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT/'outputs').resolve()):
        raise ValueError('Training output must stay inside the project outputs directory.')
    verify_binding()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    design_path = ROOT/'configs/pems_bay_validation_v1.json'
    design = read_json(design_path)
    sources = [Path(__file__).resolve(), design_path,
               *[ROOT/name for name in ('fresh_holdout_pilot.py', 'recovery_pilot.py', 'pems_bay_reader.py',
                                       'pems_bay_windows.py', 'verify_pems_bay_binding.py')],
               ROOT/'docs/pems_bay_forecaster_protocol.zh-CN.md',
               ROOT/design['dataset']['local_path'], ROOT/'outputs/pems-bay-binding/binding.json',
               ROOT/'outputs/pems-bay-binding/reader_implementation_freeze.json',
               ROOT/'outputs/pems-bay-binding/forecast_training_statistics.npz']
    run_config = {'date': '2026-10-11', 'stage': 'forecast_train_and_forecast_validation_only',
                  'protocol_id': design['protocol_id'], 'forecaster': design['forecaster'],
                  'evaluation_values_read': False, 'risk_training_executed': False,
                  'calibration_executed': False, 'held_out_evaluation_executed': False,
                  'torch_version': torch.__version__, 'numpy_version': np.__version__,
                  'device': 'cpu', 'torch_threads': 1, 'deterministic_algorithms': True,
                  'source_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in sources}}
    output.mkdir(parents=True, exist_ok=True)
    config_path = output/'config.json'
    if config_path.exists() and read_json(config_path) != run_config:
        raise ValueError('Training configuration changed; use another output directory.')
    if not config_path.exists():
        write_json(config_path, run_config)
    if (output/'results.json').exists():
        prior = read_json(output/'results.json')
        for name, digest in prior['artifact_sha256'].items():
            if sha256(output/name) != digest:
                raise ValueError(f'Completed artifact changed: {name}')
        print(json.dumps({'status': 'existing_training_verified_no_rerun', 'evaluation_values_read': False}))
        return
    started = time.perf_counter()
    reader = SealedPartitionReader(ROOT/design['dataset']['local_path'], design)
    with np.load(ROOT/'outputs/pems-bay-binding/forecast_training_statistics.npz') as stats:
        mean, scale, adjacency = (stats[name].copy() for name in ('mean', 'scale', 'adjacency'))
    intercepted = []
    # Scope the runtime guard across preparation, training and scoring.
    with guard_forecast_reads(reader, intercepted):
        train, train_origins = forecast_dataset(reader, 'forecast_train', mean, scale)
        val, val_origins = forecast_dataset(reader, 'forecast_validation', mean, scale)
        fixed_mask = artificial_mask(val.tensors[1], torch.Generator().manual_seed(991), 12)
        fingerprint = {'train_windows': len(train), 'validation_windows': len(val),
                       'train_tensor_sha256': [tensor_sha(t) for t in train.tensors],
                       'validation_tensor_sha256': [tensor_sha(t) for t in val.tensors],
                       'fixed_validation_mask_sha256': tensor_sha(fixed_mask),
                       'optimizer_updates_per_epoch': math.ceil(len(train)/128),
                       'selected_ids': reader.selected_ids}
        write_json(output/'input_fingerprints.json', fingerprint)
        np.savez_compressed(output/'origins.npz', train=train_origins, validation=val_origins)
        write_json(output/'access_log.json', {'reader': reader.access_log, 'runtime_interceptions': intercepted})
        scale_tensor = torch.tensor(scale).view(1, 1, -1)
        models, model_logs = [], []
        for seed in design['forecaster']['seeds']:
            weight_path = output/f'model_tcn_seed{seed}.pt'
            log_path = output/f'training_seed{seed}.json'
            if weight_path.exists() and log_path.exists():
                record = read_json(log_path)
                if sha256(weight_path) != record['weight_sha256'] or record['run_config_sha256'] != sha256(config_path):
                    raise ValueError('Completed seed differs from its log/configuration.')
                model = load_model(seed, adjacency, weight_path)
                print(f'Reusing completed seed {seed}', flush=True)
            elif weight_path.exists() or log_path.exists():
                raise ValueError('Incomplete seed artifacts; preserve them and use another output directory.')
            else:
                updates = [0]
                trace = []
                original_step = torch.optim.Adam.step
                original_validation = fresh_holdout_pilot.validation
                def tracked_step(optimizer, *args, **kwargs):
                    result = original_step(optimizer, *args, **kwargs)
                    updates[0] += 1
                    return result
                def tracked_validation(*args, **kwargs):
                    metrics = original_validation(*args, **kwargs)
                    trace.append({'epoch': len(trace)+1, 'updates_completed': updates[0], **metrics})
                    return metrics
                with patch.object(torch.optim.Adam, 'step', tracked_step), \
                        patch.object(fresh_holdout_pilot, 'validation', tracked_validation):
                    model, record = train_tcn(seed, train, val, scale_tensor, adjacency, output)
                expected_updates = fingerprint['optimizer_updates_per_epoch']*25
                if updates[0] != expected_updates or len(trace) != 25:
                    raise AssertionError('Frozen optimizer/validation budget not executed.')
                record.update({'optimizer_updates': updates[0], 'execution_trace': trace,
                               'weight_sha256': sha256(weight_path), 'run_config_sha256': sha256(config_path)})
                write_json(log_path, record)
            models.append(model)
            model_logs.append(record)
        predictions = np.stack([predict(model, val, fixed_mask) for model in models])
        baseline = latest(val.tensors[0], fixed_mask).numpy()[:, None, :]
        target, valid = val.tensors[2].numpy(), val.tensors[3].numpy()
        ensemble = predictions.mean(axis=0)
        summaries = {'by_seed': [error_summary(p, target, valid, scale) for p in predictions],
                     'three_seed_ensemble': error_summary(ensemble, target, valid, scale),
                     'last_visible_persistence': error_summary(np.broadcast_to(baseline, target.shape), target, valid, scale)}
        np.savez_compressed(output/'validation_predictions.npz', by_seed=predictions, ensemble=ensemble,
                            persistence=baseline, target=target, target_valid=valid, fixed_mask=fixed_mask.numpy())
        write_json(output/'access_log.json', {'reader': reader.access_log, 'runtime_interceptions': intercepted})
    if len(reader.access_log) != 2 or len(intercepted) != 2:
        raise AssertionError('Unexpected speed reads during forecast training.')
    for name, digest in run_config['source_sha256'].items():
        if sha256(ROOT/name) != digest:
            raise AssertionError(f'Frozen training source changed: {name}')
    artifacts = ['config.json', 'input_fingerprints.json', 'origins.npz', 'access_log.json', 'validation_predictions.npz']
    artifacts += [f'model_tcn_seed{seed}.pt' for seed in (42, 43, 44)]
    artifacts += [f'training_seed{seed}.json' for seed in (42, 43, 44)]
    result = {'completed_at_utc': datetime.now(timezone.utc).isoformat(),
              'stage': run_config['stage'], 'model_selection': model_logs,
              'validation': summaries, 'elapsed_seconds': time.perf_counter()-started,
              'evaluation_values_read': False, 'held_out_evaluation_executed': False,
              'risk_training_executed': False, 'calibration_executed': False,
              'artifact_sha256': {name: sha256(output/name) for name in artifacts}}
    write_json(output/'results.json', result)
    print(json.dumps({'status': 'forecast_training_completed', 'validation': summaries,
                      'evaluation_values_read': False, 'seconds': result['elapsed_seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
