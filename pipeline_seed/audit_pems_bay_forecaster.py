"""Replay trained TCN validation and independently verify selection/boundaries."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import h5py
import numpy as np
import torch
from freeze_external_validation import sha256
from verify_pems_bay_binding import main as verify_binding
from pems_bay_reader import SealedPartitionReader
from pems_bay_windows import forecast_dataset
from train_pems_bay_forecaster import (guard_forecast_reads, read_json, tensor_sha, predict, load_model,
                                     error_summary, write_json)
from recovery_pilot import artificial_mask, latest

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/pems-bay-forecaster')
    args = parser.parse_args()
    output = args.output.resolve()
    verify_binding()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config, result, fingerprints = (read_json(output/name) for name in
        ('config.json', 'results.json', 'input_fingerprints.json'))
    for name, digest in config['source_sha256'].items():
        assert sha256(ROOT/name) == digest, name
    for name, digest in result['artifact_sha256'].items():
        assert sha256(output/name) == digest, name
    design = read_json(ROOT/'configs/pems_bay_validation_v1.json')
    assert config['forecaster'] == design['forecaster']
    assert design['forecaster']['seeds'] == [42, 43, 44]
    assert design['forecaster']['epochs'] == 25 and config['deterministic_algorithms']
    assert not result['evaluation_values_read'] and not result['held_out_evaluation_executed']
    assert not result['risk_training_executed'] and not result['calibration_executed']
    reader = SealedPartitionReader(ROOT/design['dataset']['local_path'], design)
    with np.load(ROOT/'outputs/pems-bay-binding/forecast_training_statistics.npz') as saved:
        mean, scale, adjacency = [saved[key].copy() for key in ('mean', 'scale', 'adjacency')]
    interceptions = []
    with guard_forecast_reads(reader, interceptions):
        train, train_origins = forecast_dataset(reader, 'forecast_train', mean, scale)
        val, val_origins = forecast_dataset(reader, 'forecast_validation', mean, scale)
        fixed_mask = artificial_mask(val.tensors[1], torch.Generator().manual_seed(991), 12)
        assert fingerprints['train_windows'] == len(train)
        assert fingerprints['validation_windows'] == len(val)
        assert fingerprints['train_tensor_sha256'] == [tensor_sha(t) for t in train.tensors]
        assert fingerprints['validation_tensor_sha256'] == [tensor_sha(t) for t in val.tensors]
        assert fingerprints['fixed_validation_mask_sha256'] == tensor_sha(fixed_mask)
        assert fingerprints['selected_ids'] == reader.selected_ids
        with np.load(output/'origins.npz') as saved:
            assert np.array_equal(saved['train'], train_origins)
            assert np.array_equal(saved['validation'], val_origins)
        # Independent loop and sample timestamp arithmetic, not the prefix-gap helper.
        for partition, origins, data in [('forecast_train', train_origins, train),
                                         ('forecast_validation', val_origins, val)]:
            lo, hi = reader.bounds[partition]
            ts = reader.timestamps[lo:hi]
            expected = [t for t in range(12, len(ts)-2)
                        if np.all(np.diff(ts[t-12:t+3]) == np.timedelta64(5, 'm'))]
            assert np.array_equal(origins, expected)
            assert origins.min() >= 12 and origins.max()+2 < len(ts)
            assert data.tensors[0].shape == (len(origins), 12, 24)
            assert data.tensors[2].shape == (len(origins), 3, 24)
            # Input values/target positions are independently checked at the three boundaries.
            indices = [0, len(origins)//2, len(origins)-1]
            with h5py.File(reader.path, 'r') as file:
                # Guard keeps even this independent oracle to the named partition.
                order = np.argsort(reader.columns)
                raw = file['speed/block0_values'][lo:hi, reader.columns[order]][:, np.argsort(order)]
            valid = np.isfinite(raw) & (raw > 0)
            z = np.where(valid, (raw-mean)/scale, 0).astype(np.float32)
            for i in indices:
                t = int(origins[i])
                assert np.array_equal(data.tensors[0][i].numpy(), z[t-12:t])
                assert np.array_equal(data.tensors[2][i].numpy(), z[t:t+3])
                assert np.array_equal(data.tensors[1][i].numpy(), valid[t-12:t].astype(np.float32))
                assert np.array_equal(data.tensors[3][i].numpy(), valid[t:t+3].astype(np.float32))
        rebuilt = []
        selected = []
        hidden_differences = []
        updates_per_epoch = math.ceil(len(train)/128)
        for index, seed in enumerate((42, 43, 44)):
            log = read_json(output/f'training_seed{seed}.json')
            assert log == result['model_selection'][index]
            assert log['run_config_sha256'] == sha256(output/'config.json')
            assert log['weight_sha256'] == sha256(output/f'model_tcn_seed{seed}.pt')
            epochs, trace = log['epochs'], log['execution_trace']
            assert len(epochs) == len(trace) == 25
            assert [row['epoch'] for row in epochs] == list(range(1, 26))
            assert [row['updates_completed'] for row in trace] == [updates_per_epoch*i for i in range(1, 26)]
            assert log['optimizer_updates'] == 25*updates_per_epoch
            for row, observed in zip(epochs, trace):
                assert row == {key: value for key, value in observed.items() if key != 'updates_completed'}
                assert math.isfinite(row['model_mae']) and row['valid_targets'] > 0
            best = min(epochs, key=lambda row: (row['model_mae'], row['epoch']))
            assert log['selected_epoch'] == best['epoch']
            assert log['validation_mae'] == best['model_mae']
            checkpoint = torch.load(output/f'model_tcn_seed{seed}.pt', weights_only=True)
            assert checkpoint['seed'] == seed and checkpoint['best_epoch'] == best['epoch']
            assert np.array_equal(checkpoint['state_dict']['adjacency'].numpy(), adjacency)
            assert np.array_equal(checkpoint['state_dict']['two_hop'].numpy(), adjacency@adjacency)
            model = load_model(seed, adjacency, output/f'model_tcn_seed{seed}.pt')
            assert sum(p.numel() for p in model.parameters()) == log['parameters']
            p = predict(model, val, fixed_mask)
            rebuilt.append(p)
            metrics = error_summary(p, val.tensors[2].numpy(), val.tensors[3].numpy(), scale)
            assert abs(metrics['mae_mph']-log['validation_mae']) < 1e-5
            assert metrics['valid_targets'] == best['valid_targets']
            x, context, mask = val.tensors[0][:32], val.tensors[4][:32], fixed_mask[:32]
            altered = x.clone()
            altered[mask == 0] = 10000
            with torch.no_grad():
                difference = float((model(x, mask, context)-model(altered, mask, context)).abs().max())
            assert difference == 0
            hidden_differences.append(difference)
            selected.append({'seed': seed, 'epoch': best['epoch'], 'validation_mae_mph': metrics['mae_mph'],
                             'actual_updates': log['optimizer_updates']})
        try:
            reader.read_partition('evaluation')
        except PermissionError:
            pass
        else:
            raise AssertionError('Evaluation became readable during forecast audit.')
    by_seed = np.stack(rebuilt)
    ensemble = by_seed.mean(axis=0)
    target, valid = val.tensors[2].numpy(), val.tensors[3].numpy()
    baseline = latest(val.tensors[0], fixed_mask).numpy()[:, None, :]
    with np.load(output/'validation_predictions.npz') as saved:
        for name, value in [('by_seed', by_seed), ('ensemble', ensemble), ('target', target),
                            ('target_valid', valid), ('persistence', baseline), ('fixed_mask', fixed_mask.numpy())]:
            assert np.array_equal(saved[name], value), name
    expected_summary = {'by_seed': [error_summary(p, target, valid, scale) for p in by_seed],
                        'three_seed_ensemble': error_summary(ensemble, target, valid, scale),
                        'last_visible_persistence': error_summary(np.broadcast_to(baseline, target.shape), target, valid, scale)}
    assert result['validation'] == expected_summary
    # A second direct double-precision masked-error calculation checks the summary algebra.
    for name, prediction in [('three_seed_ensemble', ensemble), ('last_visible_persistence', baseline)]:
        totals = [np.sum(np.abs(prediction-target)[:, h, :].astype(np.float64)*scale[None, :]*valid[:, h, :])
                  for h in range(3)]
        count = valid.sum(dtype=np.float64)
        assert abs(sum(totals)/count-expected_summary[name]['mae_mph']) < 1e-5
    train_access = read_json(output/'access_log.json')
    assert [r['partition'] for r in train_access['reader']] == ['forecast_train', 'forecast_validation']
    assert [r['partition'] for r in train_access['runtime_interceptions']] == ['forecast_train', 'forecast_validation']
    assert len(interceptions) == 4  # two prepared tensors, two independent partition slice oracles
    audit = {'audited_at_utc': datetime.now(timezone.utc).isoformat(), 'audit_status': 'passed',
             'scope': 'three_seed_forecast_training_and_model_selection_validation_only',
             'training_source_and_all_result_artifact_hashes_verified': True,
             'input_fingerprints_origins_and_fixed_mask_rebuilt': True,
             'independent_window_and_raw_input_target_boundary_checks': True,
             'actual_25_epoch_95_updates_per_epoch_and_validation_selection_checked': True,
             'selected_models': selected, 'hidden_masked_input_max_differences': hidden_differences,
             'validation_prediction_max_rebuild_difference_mph': 0.0,
             'ensemble_persistence_metrics_and_independent_error_algebra_checked': True,
             'training_runtime_access_interceptions': train_access['runtime_interceptions'],
             'audit_runtime_access_interceptions': interceptions,
             'evaluation_values_read': False, 'held_out_evaluation_executed': False,
             'risk_training_executed': False, 'calibration_executed': False,
             'result_sha256': sha256(output/'results.json'),
             'audit_script_sha256': sha256(Path(__file__).resolve())}
    path = output/'audit.json'
    if path.exists():
        old = read_json(path)
        old_compare = {k: v for k, v in old.items() if k != 'audited_at_utc'}
        new_compare = {k: v for k, v in audit.items() if k != 'audited_at_utc'}
        if old_compare != new_compare:
            raise ValueError('Existing audit differs; preserve it and use an explicit new version.')
    else:
        write_json(path, audit)
    print(json.dumps({'audit_status': 'passed', 'selected_models': selected,
                      'evaluation_values_read': False, 'hidden_masked_input_max_differences': hidden_differences}, indent=2))


if __name__ == '__main__':
    main()
