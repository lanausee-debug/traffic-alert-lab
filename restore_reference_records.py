"""Restore three exact frozen JSON records from compressed repository files."""
from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def restore():
    records = json.loads((ROOT/'reference_records/manifest.json').read_text(encoding='utf-8'))
    for row in records:
        source = (ROOT/row['compressed_path']).resolve()
        destination = (ROOT/row['original_path']).resolve()
        source.relative_to(ROOT)
        destination.relative_to(ROOT)
        packed = source.read_bytes()
        if hashlib.sha256(packed).hexdigest() != row['compressed_sha256']:
            raise ValueError('Compressed reference changed: '+row['compressed_path'])
        data = gzip.decompress(packed)
        if len(data) != row['original_bytes'] or hashlib.sha256(data).hexdigest() != row['original_sha256']:
            raise ValueError('Reference record checksum mismatch: '+row['original_path'])
        if destination.exists():
            if destination.read_bytes() != data:
                raise ValueError('Refusing to replace a changed existing record: '+row['original_path'])
        else:
            destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open('xb') as stream:
                stream.write(data)
    return len(records)


if __name__ == '__main__':
    print(json.dumps({'exact_reference_records_restored_or_verified':restore(),'traffic_data_decoded':False}))
