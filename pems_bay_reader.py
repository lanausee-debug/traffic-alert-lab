"""Metadata-first, partition-bounded PEMS-BAY reader. Evaluation stays sealed."""
from __future__ import annotations

import hashlib
from pathlib import Path
import h5py
import numpy as np


class SealedPartitionReader:
    """A cooperative pipeline guard, not OS access control over the raw file.

    The current implementation never permits evaluation values. A later audited
    execution seal is required before adding evaluation access to the pipeline.
    """

    def __init__(self, path, config):
        self.path = Path(path)
        self.config = config
        self.access_log = []
        with h5py.File(self.path, 'r') as file:
            required = ['speed/axis0', 'speed/axis1', 'speed/block0_items', 'speed/block0_values']
            if any(key not in file for key in required):
                raise ValueError('Unsupported HDF5 layout: require fixed-format numeric speed block.')
            axis = file['speed/axis0'][:]
            items = file['speed/block0_items'][:]
            times = file['speed/axis1'][:]
            values = file['speed/block0_values']
            if axis.ndim != 1 or not np.array_equal(axis, items):
                raise ValueError('Column axis and value-block items do not agree.')
            if times.ndim != 1 or times.dtype.kind != 'i':
                raise ValueError('Require integer nanosecond timestamp metadata.')
            if values.shape != (len(times), len(axis)) or values.dtype.kind != 'f':
                raise ValueError('Require time-by-node floating speed matrix.')
            self.value_shape, self.value_dtype = values.shape, str(values.dtype)
            self.value_chunks = values.chunks
            # Header shape/dtype access above does not read the speed dataset.
        # This benchmark uses numeric IDs; reject ambiguous bytes/object IDs.
        if axis.dtype.kind not in 'iu':
            raise ValueError('Require numeric station-ID metadata; do not unpickle object IDs.')
        self.ids = [str(item).strip() for item in axis]
        if any(not item for item in self.ids) or len(set(self.ids)) != len(self.ids):
            raise ValueError('Empty/duplicate canonical station IDs.')
        self.timestamps = times.astype('datetime64[ns]')
        if len(times) < 2 or np.isnat(self.timestamps).any() or np.any(np.diff(times) <= 0):
            raise ValueError('Timestamp axis must be finite and strictly increasing.')
        self.d0 = self.timestamps[0].astype('datetime64[D]')
        # Detect incorrect timestamp units without deriving anything from speed.
        if not np.datetime64('2000-01-01') <= self.d0 < np.datetime64('2100-01-01'):
            raise ValueError('Timestamp unit/epoch is inconsistent with this benchmark.')
        self.end_exclusive = self.timestamps[-1] + np.timedelta64(5, 'm')
        if self.end_exclusive < self.d0 + np.timedelta64(config['dataset']['minimum_calendar_days'], 'D'):
            raise ValueError('Insufficient calendar span for the fixed partition plan.')
        if np.any(np.diff(times) % (5*60*10**9)) or np.any(times % (5*60*10**9)):
            raise ValueError('Timestamps must lie on the native five-minute lattice.')
        salt = config['dataset']['station_selection_salt']
        ranked = sorted(self.ids, key=lambda item: (hashlib.sha256(f'{salt}|{item}'.encode('utf-8')).hexdigest(), item))
        count = config['dataset']['station_count']
        if len(ranked) < count:
            raise ValueError('Too few stations for the frozen plan.')
        self.selected_ids = sorted(ranked[:count])
        index = {item: i for i, item in enumerate(self.ids)}
        self.columns = np.array([index[item] for item in self.selected_ids], dtype=np.int64)
        self.bounds = {}
        for name, (start_day, end_day) in config['partitions_days_half_open'].items():
            start = self.d0 + np.timedelta64(start_day, 'D')
            end = self.end_exclusive if end_day is None else self.d0 + np.timedelta64(end_day, 'D')
            lo, hi = (int(np.searchsorted(self.timestamps, timestamp)) for timestamp in (start, end))
            if hi <= lo:
                raise ValueError(f'Empty partition: {name}')
            self.bounds[name] = (lo, hi)

    def metadata(self):
        deltas = np.diff(self.timestamps)
        gap = deltas != np.timedelta64(5, 'm')
        return {'shape': list(self.value_shape), 'value_dtype': self.value_dtype,
                'value_chunks': self.value_chunks,
                'timestamp_first': str(self.timestamps[0]), 'timestamp_last': str(self.timestamps[-1]),
                'end_exclusive': str(self.end_exclusive), 'day_zero': str(self.d0),
                'native_time_axis_no_timezone_conversion': True,
                'calendar_days_touched': int((self.timestamps[-1].astype('datetime64[D]')-self.d0)/np.timedelta64(1, 'D'))+1,
                'non_five_minute_gaps': int(gap.sum()),
                'gap_metadata': [{'left': str(self.timestamps[i]), 'right': str(self.timestamps[i+1]),
                                  'minutes': int(deltas[i]/np.timedelta64(1, 'm'))} for i in np.flatnonzero(gap)],
                'selected_ids': self.selected_ids, 'selected_raw_column_indices': self.columns.tolist(),
                'partitions': {name: {'row_start': lo, 'row_stop_exclusive': hi,
                                     'first_timestamp': str(self.timestamps[lo]),
                                     'last_timestamp': str(self.timestamps[hi-1]), 'rows': hi-lo}
                               for name, (lo, hi) in self.bounds.items()},
                'speed_value_reads': len(self.access_log), 'evaluation_values_accessible': False}

    def _authorize(self, partition, start, stop):
        if partition not in self.bounds:
            raise PermissionError('Only declared partitions can be read.')
        if partition == 'evaluation':
            raise PermissionError('Evaluation speed values are sealed; execution freeze not implemented yet.')
        lo, hi = self.bounds[partition]
        if not (lo <= start < stop <= hi):
            raise PermissionError('Requested rows cross the declared partition.')
        if stop > self.bounds['evaluation'][0]:
            raise PermissionError('Read intersects the sealed evaluation tail.')

    def _read_values(self, partition, start, stop):
        self._authorize(partition, start, stop)  # before opening/indexing the dataset
        order = np.argsort(self.columns)
        restore = np.argsort(order)
        with h5py.File(self.path, 'r') as file:
            dataset = file['speed/block0_values']
            raw = dataset[start:stop, self.columns[order]][:, restore]
        self.access_log.append({'partition': partition, 'dataset': 'speed/block0_values',
                                'row_start': int(start), 'row_stop_exclusive': int(stop),
                                'columns': self.columns.tolist(), 'logical_values': int(raw.size)})
        return raw

    def read_partition(self, partition):
        if partition not in self.bounds:
            raise PermissionError('Unknown partition.')
        lo, hi = self.bounds[partition]
        raw = self._read_values(partition, lo, hi)
        return raw, self.timestamps[lo:hi].copy()


def training_statistics(reader):
    """Read ONLY forecast training speed; derive the fixed graph and groups."""
    raw, timestamps = reader.read_partition('forecast_train')
    valid = np.isfinite(raw) & (raw > 0)
    availability = valid.mean(axis=0)
    floor = reader.config['dataset']['training_availability_floor']
    quality = {'by_station': dict(zip(reader.selected_ids, availability.tolist())),
               'minimum': float(availability.min()), 'required_minimum': floor,
               'passed': bool(np.all(availability >= floor)),
               'station_replacement_permitted': False, 'training_rows': len(raw)}
    if not quality['passed']:
        return quality, None
    counts = valid.sum(axis=0)
    mean64 = np.where(valid, raw, 0).sum(axis=0)/counts
    variance = np.where(valid, (raw-mean64)**2, 0).sum(axis=0)/counts
    mean = mean64.astype(np.float32)
    scale = np.maximum(np.sqrt(variance), 1e-3).astype(np.float32)
    z = np.where(valid, (raw-mean)/scale, 0).astype(np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = np.maximum(np.nan_to_num(np.corrcoef(z.T)), 0)
    np.fill_diagonal(corr, 0)
    count = len(reader.selected_ids)
    adjacency = np.zeros((count, count), dtype=np.float32)
    ranked_neighbors = []
    for i in range(count):
        ranked = sorted((j for j in range(count) if j != i), key=lambda j: (-corr[i, j], reader.selected_ids[j]))
        ranked_neighbors.append(ranked)
        top = ranked[:min(4, count-1)]
        adjacency[i, top] = corr[i, top]
        adjacency[i, i] = 1
    adjacency /= adjacency.sum(axis=1, keepdims=True)
    groups = [{'center_id': reader.selected_ids[i], 'nodes': [i]+ranked_neighbors[i][:2],
               'sensor_ids': [reader.selected_ids[j] for j in [i]+ranked_neighbors[i][:2]]}
              for i in range(min(12, count))]
    statistics = {'mean': mean, 'scale': scale, 'correlation': corr, 'adjacency': adjacency,
                  'groups': groups, 'training_first_timestamp': str(timestamps[0]),
                  'training_last_timestamp': str(timestamps[-1])}
    return quality, statistics
