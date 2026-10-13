"""Audit real training-only access plus synthetic leakage/guard failure cases."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import h5py
import numpy as np
from freeze_external_validation import sha256, verify
from pems_bay_reader import SealedPartitionReader, training_statistics
from pems_bay_windows import contiguous_origins, forecast_arrays

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT/'outputs/pems-bay-binding'


def expect_denied(operation):
    try:
        operation()
    except PermissionError:
        return
    raise AssertionError('Sealed or crossing read unexpectedly succeeded.')


def synthetic_checks(config):
    """Synthetic arrays have no relationship to either real traffic dataset."""
    cfg = copy.deepcopy(config)
    cfg['dataset']['station_count'] = 3
    ids = np.array([400100, 400999, 400020, 400333, 400002], dtype=np.int64)
    times = np.arange(np.datetime64('2017-01-01T00:00', 'ns'),
                      np.datetime64('2017-05-09T00:00', 'ns'), np.timedelta64(5, 'm'))
    x = np.arange(len(times), dtype=np.float64)
    raw = np.stack([45+i+np.sin(x/(20+i))*3 for i in range(len(ids))], axis=1)
    with tempfile.TemporaryDirectory(prefix='pems-reader-audit-') as directory:
        path = Path(directory)/'synthetic.h5'
        def fixture(timestamp_values=times, block_ids=ids):
            with h5py.File(path, 'w') as file:
                group = file.create_group('speed')
                group.create_dataset('axis0', data=ids)
                group.create_dataset('axis1', data=timestamp_values.astype(np.int64))
                group.create_dataset('block0_items', data=block_ids)
                group.create_dataset('block0_values', data=raw)
        fixture()
        reader = SealedPartitionReader(path, cfg)
        assert reader.access_log == []
        selected = sorted(sorted([str(i) for i in ids], key=lambda i: (
            hashlib.sha256((cfg['dataset']['station_selection_salt']+'|'+i).encode()).digest(), i))[:3])
        assert reader.selected_ids == selected
        partition, _ = reader.read_partition('forecast_train')
        expected_columns = [list(ids).index(int(i)) for i in selected]
        assert np.array_equal(partition, raw[:42*288, expected_columns])
        before = len(reader.access_log)
        expect_denied(lambda: reader.read_partition('evaluation'))
        expect_denied(lambda: reader.read_partition('whole_file'))
        expect_denied(lambda: reader._read_values('forecast_train', -1, 10))
        expect_denied(lambda: reader._read_values('forecast_train', 0, 42*288+1))
        expect_denied(lambda: reader._read_values('calibration', 84*288, 98*288+1))
        assert len(reader.access_log) == before
        _, first = training_statistics(reader)
        tail = reader.bounds['evaluation'][0]
        with h5py.File(path, 'r+') as file:
            file['speed/block0_values'][tail:, :] = -999999
        changed = SealedPartitionReader(path, cfg)
        quality, second = training_statistics(changed)
        assert quality['passed'] and changed.selected_ids == selected
        for key in ('mean', 'scale', 'correlation', 'adjacency'):
            assert np.array_equal(first[key], second[key])
        assert first['groups'] == second['groups']
        with h5py.File(path, 'r+') as file:
            file['speed/block0_values'][:42*288, expected_columns[0]] = 0
        bad_quality, absent = training_statistics(SealedPartitionReader(path, cfg))
        assert not bad_quality['passed'] and absent is None
        assert not bad_quality['station_replacement_permitted']
        fixture(block_ids=ids[::-1])
        try:
            SealedPartitionReader(path, cfg)
        except ValueError:
            pass
        else:
            raise AssertionError('Misaligned value-block columns accepted.')
        duplicate_times = times.copy()
        duplicate_times[1] = duplicate_times[0]
        fixture(timestamp_values=duplicate_times)
        try:
            SealedPartitionReader(path, cfg)
        except ValueError:
            pass
        else:
            raise AssertionError('Duplicate timestamp accepted.')
    return {'synthetic_id_hash_selection_and_unsorted_column_restoration': True,
            'evaluation_unknown_partition_negative_and_cross_partition_reads_denied': True,
            'evaluation_tail_poison_does_not_change_train_statistics_graph_or_groups': True,
            'training_quality_failure_does_not_replace_stations': True,
            'misaligned_columns_and_duplicate_timestamps_rejected': True}


def main():
    verify()
    config = json.loads((ROOT/'configs/pems_bay_validation_v1.json').read_text(encoding='utf-8'))
    binding = json.loads((OUTPUT/'binding.json').read_text(encoding='utf-8'))
    for name, digest in binding['source_sha256'].items():
        assert sha256(ROOT/name) == digest, name
    for name, digest in binding['artifact_sha256'].items():
        assert sha256(OUTPUT/name) == digest, name
    data = ROOT/config['dataset']['local_path']
    intercepted = []
    original = h5py.Dataset.__getitem__
    allowed_stop = 42*288  # real source first 42 native days has no metadata gap
    def guarded_getitem(dataset, key, *args, **kwargs):
        if dataset.name == '/speed/block0_values':
            assert isinstance(key, tuple) and isinstance(key[0], slice)
            rows = key[0]
            assert rows.start == 0 and rows.stop == allowed_stop and rows.step in (None, 1)
            intercepted.append({'row_start': rows.start, 'row_stop_exclusive': rows.stop,
                                'columns': np.asarray(key[1]).tolist()})
        return original(dataset, key, *args, **kwargs)
    with patch.object(h5py.Dataset, '__getitem__', guarded_getitem):
        reader = SealedPartitionReader(data, config)
        assert len(intercepted) == 0
        metadata = reader.metadata()
        saved_metadata = json.loads((OUTPUT/'metadata.json').read_text(encoding='utf-8'))
        assert metadata == saved_metadata
        quality, statistics = training_statistics(reader)
        assert len(intercepted) == 1 and quality['passed']
        before = len(intercepted)
        expect_denied(lambda: reader.read_partition('evaluation'))
        expect_denied(lambda: reader._read_values('calibration', reader.bounds['calibration'][0],
                                                reader.bounds['evaluation'][0]+1))
        assert len(intercepted) == before
    salt = config['dataset']['station_selection_salt']
    independent = sorted(sorted(reader.ids, key=lambda i: (hashlib.sha256((salt+'|'+i).encode()).digest(), i))[:24])
    assert independent == reader.selected_ids
    with np.load(OUTPUT/'forecast_training_statistics.npz') as saved:
        for key in ('mean', 'scale', 'correlation', 'adjacency'):
            assert np.array_equal(saved[key], statistics[key]), key
    assert statistics['groups'] == json.loads((OUTPUT/'fault_groups.json').read_text(encoding='utf-8'))
    assert np.all(statistics['adjacency'].diagonal() > 0)
    assert np.allclose(statistics['adjacency'].sum(axis=1), 1)
    assert all(len(set(group['nodes'])) == 3 for group in statistics['groups'])
    checks = synthetic_checks(config)
    # Independent loop oracle verifies exclusion at split edges and a clock gap.
    clock = np.arange(np.datetime64('2017-03-12T00:00', 'ns'),
                      np.datetime64('2017-03-12T08:00', 'ns'), np.timedelta64(5, 'm'))
    clock = np.delete(clock, np.arange(24, 36))
    for horizon in (3, 5):
        oracle = [t for t in range(12, len(clock)-horizon+1)
                  if all(np.diff(clock[t-12:t+horizon]) == np.timedelta64(5, 'm'))]
        assert np.array_equal(contiguous_origins(clock, 12, horizon), oracle)
    fake_raw = np.arange(len(clock)*3, dtype=np.float64).reshape(-1, 3)+10
    fake_mean, fake_scale = np.ones(3)*20, np.ones(3)*5
    origins, arrays = forecast_arrays(fake_raw, clock, fake_mean, fake_scale)
    for i, t in enumerate(origins):
        assert np.array_equal(arrays[0][i], ((fake_raw[t-12:t]-fake_mean)/fake_scale).astype(np.float32))
        assert np.array_equal(arrays[2][i], ((fake_raw[t:t+3]-fake_mean)/fake_scale).astype(np.float32))
    t = int(origins[0])
    modified = fake_raw.copy()
    modified[t:] = 12345
    other_origins, other_arrays = forecast_arrays(modified, clock, fake_mean, fake_scale)
    assert np.array_equal(origins, other_origins)
    assert np.array_equal(arrays[0][0], other_arrays[0][0])
    assert np.array_equal(arrays[4][0], other_arrays[4][0])
    expect_denied(lambda: __import__('pems_bay_windows').forecast_dataset(reader, 'evaluation',
                  statistics['mean'], statistics['scale']))
    checks['local_window_edges_gap_exclusion_and_future_input_invariance'] = True
    window_counts = {}
    for name, (lo, hi) in reader.bounds.items():
        horizon = 3 if name.startswith('forecast_') else 5
        window_counts[name] = {'metadata_only_contiguous_origins': len(contiguous_origins(reader.timestamps[lo:hi], 12, horizon)),
                               'history': 12, 'confirmation_horizon': horizon}
    audit = {'audited_at_utc': datetime.now(timezone.utc).isoformat(),
             'audit_status': 'passed', 'scope': 'source_binding_metadata_training_statistics_and_sealed_reader',
             'real_speed_dataset_getitem_interceptions': intercepted,
             'real_metadata_speed_reads': 0, 'evaluation_values_read': False,
             'actual_model_or_calibration_execution_audited': False,
             'validation_executed': False, 'binding_hashes_verified': True,
             'independent_hash_station_selection_checked': True,
             'saved_training_statistics_and_fault_groups_rebuilt_exactly': True,
             'synthetic_checks': checks,
             'limitations': 'Logical pipeline access guard; raw-file permissions do not cryptographically seal values.',
             'audit_script_sha256': sha256(Path(__file__).resolve()),
             'binding_sha256': sha256(OUTPUT/'binding.json'),
             'reader_sha256': sha256(ROOT/'pems_bay_reader.py')}
    audit['windows_script_sha256'] = sha256(ROOT/'pems_bay_windows.py')
    audit['metadata_window_counts'] = window_counts
    (OUTPUT/'reader_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'audit_status': 'passed', 'real_speed_reads': len(intercepted),
                      'evaluation_values_read': False, 'synthetic_checks': checks}, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
