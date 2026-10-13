"""Local partition windows; cannot join histories/targets across a gap or split."""
from __future__ import annotations
import numpy as np


def contiguous_origins(timestamps, history=12, horizon=3):
    timestamps = np.asarray(timestamps)
    if history < 1 or horizon < 1 or timestamps.ndim != 1:
        raise ValueError('Invalid local window definition.')
    origins = np.arange(history, len(timestamps)-horizon+1, dtype=np.int64)
    bad_prefix = np.r_[0, np.cumsum(np.diff(timestamps) != np.timedelta64(5, 'm'))]
    # A window covers timestamp indices [t-history, t+horizon-1].
    good = bad_prefix[origins+horizon-1] == bad_prefix[origins-history]
    return origins[good]


def calendar_features(timestamps):
    timestamps = np.asarray(timestamps)
    day = timestamps.astype('datetime64[D]')
    minute = (timestamps-day).astype('timedelta64[m]').astype(np.float32)
    weekday = (day.astype(np.int64)+3) % 7
    return np.stack([np.sin(2*np.pi*minute/1440), np.cos(2*np.pi*minute/1440),
                     np.sin(2*np.pi*weekday/7), np.cos(2*np.pi*weekday/7)], axis=1).astype(np.float32)


def forecast_arrays(raw, timestamps, mean, scale):
    raw, timestamps = np.asarray(raw), np.asarray(timestamps)
    if raw.ndim != 2 or len(raw) != len(timestamps) or raw.shape[1] != len(mean) or np.any(scale <= 0):
        raise ValueError('Shape/normalization mismatch.')
    origins = contiguous_origins(timestamps, 12, 3)
    if len(origins) == 0:
        raise ValueError('No contiguous forecast windows in partition.')
    valid = (np.isfinite(raw) & (raw > 0)).astype(np.float32)
    normalized = np.where(valid > 0, (raw-mean)/scale, 0).astype(np.float32)
    past = origins[:, None]+np.arange(-12, 0)[None]
    future = origins[:, None]+np.arange(3)[None]
    return origins, (normalized[past], valid[past], normalized[future], valid[future],
                     calendar_features(timestamps[origins]))


def forecast_dataset(reader, partition, mean, scale):
    """An adapter for the existing TCN trainer, restricted to its two splits."""
    if partition not in ('forecast_train', 'forecast_validation'):
        raise PermissionError('Forecaster training adapter only accepts its declared train/validation partitions.')
    import torch
    from torch.utils.data import TensorDataset
    raw, timestamps = reader.read_partition(partition)
    origins, arrays = forecast_arrays(raw, timestamps, mean, scale)
    return TensorDataset(*(torch.from_numpy(array) for array in arrays)), origins
