"""Freeze the external validation DESIGN; this does not execute an experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / 'configs/pems_bay_validation_v1.json'
OUTPUT = ROOT / 'outputs/pems-bay-validation-plan'
MANIFEST = OUTPUT / 'design_freeze.json'


def sha256(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256')
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check_design(config):
    """Check decision-critical design consistency without reading new data."""
    if config['status'] != 'plan_only_not_executed':
        raise ValueError('Design cannot be marked as an executed validation.')
    segments = config['partitions_days_half_open']
    pairs = list(segments.values())
    if pairs[0][0] != 0 or pairs[-1] != [98, None]:
        raise ValueError('Invalid evaluation boundary.')
    for previous, following in zip(pairs, pairs[1:]):
        if previous[1] != following[0] or previous[0] >= previous[1]:
            raise ValueError('Partitions must be ordered and disjoint.')
    if config['dataset']['minimum_calendar_days'] - pairs[-1][0] < 28:
        raise ValueError('Require at least 28 evaluation calendar days.')
    evaluation = config['evaluation']
    if evaluation['primary_references'] != ['global_calibration_fine', 'group_calibration_fine']:
        raise ValueError('Primary references cannot be replaced by a weaker selected baseline.')
    if evaluation['primary_family_size'] != len(evaluation['primary_references']):
        raise ValueError('Primary family size mismatch.')
    if evaluation['primary_interval_confidence'] != 1 - .05/evaluation['primary_family_size']:
        raise ValueError('Primary interval multiplicity mismatch.')
    if evaluation['evaluation_grid_scan'] or config['evaluation_value_access_before_selection_freeze']:
        raise ValueError('Evaluation must remain sealed until selections are frozen.')
    identifiers = [method['id'] for method in config['methods']]
    if len(identifiers) != 7 or len(set(identifiers)) != 7 or 'onset_mlp' not in identifiers:
        raise ValueError('Invalid fixed method family.')
    if config['risk']['feature_dimension'] != 27 + config['dataset']['station_count']:
        raise ValueError('Risk feature dimension mismatch.')
    if config['risk']['mlp_checkpoints'][-1] != config['risk']['mlp_updates']:
        raise ValueError('MLP selection schedule mismatch.')
    calibration = config['calibration']
    fine = calibration['probability_fine']
    # Integer thousandths avoid floating grid membership ambiguity.
    fine_points = set(range(round(fine['start']*1000), round(fine['stop_inclusive']*1000)+1,
                            round(fine['step']*1000)))
    coarse = calibration['probability_coarse']
    if not all(round(t*1000) in fine_points for t in coarse if t != fine['off']):
        raise ValueError('Coarse probability grid must be contained in fine grid.')
    if coarse[-1] != fine['off'] or calibration['budget_relaxation'] != 0:
        raise ValueError('Missing off setting or changed budget.')


def sources():
    return [CONFIG, Path(__file__).resolve(),
            ROOT/'docs/thesis_contribution_boundary.zh-CN.md',
            ROOT/'docs/pems_bay_validation_protocol.zh-CN.md',
            *[ROOT/name for name in ('fresh_holdout_pilot.py', 'recovery_pilot.py',
                'congestion_alert_pilot.py', 'congestion_risk_pilot.py',
                'congestion_solver_audit.py', 'congestion_density_pilot.py')],
            *[ROOT/'outputs/congestion-fine-grid-pilot'/name for name in
              ('config.json', 'results.json', 'audit.json', 'selection.json')],
            ROOT/'docs/congestion_stage8_findings.zh-CN.md']


def verify():
    manifest = read_json(MANIFEST)
    check_design(read_json(CONFIG))
    differences = [name for name, digest in manifest['source_sha256'].items()
                   if not (ROOT/name).is_file() or sha256(ROOT/name) != digest]
    if differences:
        raise ValueError('Frozen sources changed: ' + ', '.join(differences))
    if manifest['validation_executed'] or manifest['data_access_audit_passed']:
        raise ValueError('A design freeze cannot certify execution or data isolation.')
    if manifest['dataset_present_at_freeze']:
        raise ValueError('This v1 expects design freezing before local data acquisition.')
    return {'design_integrity': 'passed', 'frozen_files': len(manifest['source_sha256']),
            'validation_executed': False, 'data_access_audit_passed': False,
            'next_gate': 'dataset_binding_and_partition_reader_implementation_audit'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='Read-only integrity verification')
    args = parser.parse_args()
    if not args.verify and not MANIFEST.exists():
        config = read_json(CONFIG)
        check_design(config)
        dataset_present = (ROOT/config['dataset']['local_path']).exists()
        if dataset_present:
            raise ValueError('Unexpected new-data file: audit access history before freezing.')
        audit = read_json(ROOT/'outputs/congestion-fine-grid-pilot/audit.json')
        if audit['audit_status'] != 'passed':
            raise ValueError('Existing exploratory evidence audit is not passed.')
        manifest = {'protocol_id': config['protocol_id'],
                    'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
                    'freeze_kind': 'local_design_only', 'validation_executed': False,
                    'data_access_audit_passed': False,
                    'dataset_present_at_freeze': dataset_present,
                    'source_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in sources()}}
        OUTPUT.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    result = verify()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
