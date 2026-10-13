"""Read-only verification of the frozen reader milestone and data byte identity."""
import json
from pathlib import Path
from freeze_external_validation import sha256, verify

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT/'outputs/pems-bay-binding'


def main():
    verify()
    manifest = json.loads((OUTPUT/'reader_implementation_freeze.json').read_text(encoding='utf-8'))
    binding = json.loads((OUTPUT/'binding.json').read_text(encoding='utf-8'))
    for entries in (manifest['source_sha256'], binding['source_sha256']):
        for name, digest in entries.items():
            if sha256(ROOT/name) != digest:
                raise ValueError(f'Frozen source changed: {name}')
    for name, digest in binding['artifact_sha256'].items():
        if sha256(OUTPUT/name) != digest:
            raise ValueError(f'Bound artifact changed: {name}')
    audit = json.loads((OUTPUT/'reader_audit.json').read_text(encoding='utf-8'))
    if audit['audit_status'] != 'passed' or manifest['evaluation_values_read'] or manifest['validation_executed']:
        raise ValueError('Unexpected execution state.')
    print(json.dumps({'frozen_reader_integrity': 'passed', 'source_data_bytes_verified': True,
                      'frozen_files': len(manifest['source_sha256']),
                      'evaluation_speed_values_deserialized_by_this_verifier': False,
                      'model_training_executed': False, 'validation_executed': False}, indent=2))


if __name__ == '__main__':
    main()
