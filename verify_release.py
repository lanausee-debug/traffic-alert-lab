"""Standard-library release file and aggregate explorer checks, no traffic reads."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
from restore_reference_records import restore

ROOT = Path(__file__).resolve().parent


def main():
    restored = restore()
    manifest = json.loads((ROOT / 'release_manifest.json').read_text(encoding='utf-8'))
    for name, meta in manifest['files'].items():
        p = (ROOT / name).resolve()
        p.relative_to(ROOT)
        data = p.read_bytes()
        if len(data) != meta['bytes'] or hashlib.sha256(data).hexdigest() != meta['sha256']:
            raise ValueError('Changed release file: ' + name)
    source = json.loads((ROOT / 'outputs/pems-bay-evaluation/results.json').read_text(encoding='utf-8'))
    html = (ROOT / 'demo/index.html').read_text(encoding='utf-8')
    aggregate = json.loads(re.search(r'<script id="result-data" type="application/json">(.*?)</script>', html, re.S).group(1))
    assert aggregate['events'] == 673 and aggregate['nodes'] == 24 and aggregate['days'] == 83
    assert set(aggregate['methods']) == set(source['evaluation'])
    for method, entry in aggregate['methods'].items():
        for scenario, metrics in entry['scenarios'].items():
            for name, value in metrics.items():
                assert value == source['evaluation'][method]['fine'][scenario][name], (method, scenario, name)
    assert source['verdict']['primary_metric_supported'] is False
    assert 'joint primary test did not pass' in html
    assert not any(s in html for s in ('fetch(', 'XMLHttpRequest', '<script src=', '__RESULT_DATA__'))
    assert manifest['raw_data_or_weights_included'] is False
    assert manifest['actual_station_time_series_included'] is False
    print(json.dumps({'release_files_verified': len(manifest['files']), 'aggregate_methods': len(aggregate['methods']),
                      'fine_scenario_settings': 21, 'primary_metric_supported': False,
                      'exact_reference_records_restored_or_verified': restored,
                      'raw_traffic_data_read': False, 'research_training_performed': False}))


if __name__ == '__main__':
    main()
