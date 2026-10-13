"""Bind source identity/metadata and forecast-training-only statistics to v1."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from freeze_external_validation import verify, sha256
from pems_bay_reader import SealedPartitionReader, training_statistics

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT/'outputs/pems-bay-binding'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    verify()
    config_path = ROOT/'configs/pems_bay_validation_v1.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    data = ROOT/config['dataset']['local_path']
    provenance_path = ROOT/'data/pems-bay-provenance.json'
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    if sha256(data) != provenance['sha256'] or data.stat().st_size != provenance['bytes']:
        raise ValueError('Source byte identity differs from the acquisition record.')
    if (OUTPUT/'binding.json').exists():
        saved = json.loads((OUTPUT/'binding.json').read_text(encoding='utf-8'))
        for name, digest in saved['source_sha256'].items():
            if sha256(ROOT/name) != digest:
                raise ValueError(f'Bound source changed: {name}')
        for name, digest in saved['artifact_sha256'].items():
            if sha256(OUTPUT/name) != digest:
                raise ValueError(f'Bound artifact changed: {name}')
        print(json.dumps({'status': 'existing_binding_verified', 'evaluation_values_read': False}))
        return
    reader = SealedPartitionReader(data, config)
    metadata = reader.metadata()
    if metadata['shape'][1] != 325:
        raise ValueError('Source does not have the expected benchmark 325-node header.')
    if metadata['speed_value_reads'] != 0:
        raise AssertionError('Metadata inspection unexpectedly read speed values.')
    quality, statistics = training_statistics(reader)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT/'metadata.json', metadata)
    write_json(OUTPUT/'training_quality.json', quality)
    write_json(OUTPUT/'access_log.json', reader.access_log)
    artifacts = ['metadata.json', 'training_quality.json', 'access_log.json']
    if statistics is not None:
        np.savez_compressed(OUTPUT/'forecast_training_statistics.npz',
                            **{key: statistics[key] for key in ('mean', 'scale', 'correlation', 'adjacency')})
        write_json(OUTPUT/'fault_groups.json', statistics['groups'])
        artifacts += ['forecast_training_statistics.npz', 'fault_groups.json']
    sources = [config_path, data, provenance_path, Path(__file__).resolve(), ROOT/'pems_bay_reader.py',
               ROOT/'outputs/pems-bay-validation-plan/design_freeze.json',
               ROOT/'data/source-metadata/pems-bay-discovery.json']
    binding = {'protocol_id': config['protocol_id'], 'bound_at_utc': datetime.now(timezone.utc).isoformat(),
               'status': 'ready_for_pre_evaluation_training' if quality['passed'] else 'infeasible_station_training_quality',
               'dataset_sha256': provenance['sha256'], 'speed_unit': 'mph',
               'unit_evidence': {'source': 'https://iclr.cc/media/iclr-2022/Slides/5891.pdf',
                                 'page_one_based': 12, 'claim': 'PEMS-BAY average speed measured in mph',
                                 'method': 'primary research dataset description, not inference from evaluation speeds'},
               'metadata_speed_value_reads': 0, 'speed_reads': reader.access_log,
               'evaluation_values_read': False, 'validation_executed': False,
               'source_sha256': {path.relative_to(ROOT).as_posix(): sha256(path) for path in sources},
               'artifact_sha256': {name: sha256(OUTPUT/name) for name in artifacts}}
    write_json(OUTPUT/'binding.json', binding)
    print(json.dumps({'status': binding['status'], 'shape': metadata['shape'],
                      'first': metadata['timestamp_first'], 'last': metadata['timestamp_last'],
                      'training_minimum_availability': quality['minimum'],
                      'speed_value_reads': reader.access_log, 'evaluation_values_read': False}, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
