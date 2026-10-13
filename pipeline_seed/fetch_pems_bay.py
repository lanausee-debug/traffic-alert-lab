"""Download bytes of the bound public PEMS-BAY HDF5, without loading speed values."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
EXPECTED_SHA = '65d69fb0a2323dba9867179eb7af47c8b814186bc459ff0a4937d21614153c8f'
EXPECTED_BYTES = 135930936
REVISION = 'c858397fb381d82c3d55ad0a81beb51a9bb9bdaa'
URLS = [
    'https://drive.usercontent.google.com/download?id=1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq&export=download&confirm=t',
    f'https://huggingface.co/datasets/jimmygao3218/PEMSBAY/resolve/{REVISION}/PEMS-BAY.h5',
]
TARGET = ROOT/'data/raw/pems-bay.h5'
PROVENANCE = ROOT/'data/pems-bay-provenance.json'


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    from freeze_external_validation import verify
    verify()
    if TARGET.exists():
        if TARGET.stat().st_size != EXPECTED_BYTES or sha256(TARGET) != EXPECTED_SHA:
            raise ValueError('Existing file does not match inspected distribution; do not overwrite.')
        if not PROVENANCE.exists():
            raise ValueError('Existing data has no provenance; audit acquisition before continuing.')
        print(json.dumps({'status': 'existing_verified', 'bytes': EXPECTED_BYTES, 'sha256': EXPECTED_SHA}))
        return
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    part = TARGET.with_suffix('.download')
    attempts = []
    used_url = None
    for url in URLS:
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'traffic-alert-validation/1.0'})
            started = time.monotonic()
            print(f'Trying public distribution: {url}', flush=True)
            with urllib.request.urlopen(request, timeout=30) as response, part.open('wb') as output:
                count = 0
                next_progress = 8*1024*1024
                while chunk := response.read(1024*1024):
                    if count == 0 and not chunk.startswith(b'\x89HDF\r\n\x1a\n'):
                        raise ValueError('Response is not HDF5 (possibly a confirmation page).')
                    output.write(chunk)
                    count += len(chunk)
                    if count > EXPECTED_BYTES:
                        raise ValueError('Distribution exceeds bound expected size.')
                    if count >= next_progress:
                        print(f'Downloaded {count//(1024*1024)} MiB', flush=True)
                        next_progress += 8*1024*1024
                    if time.monotonic()-started > 600:
                        raise TimeoutError('Transfer exceeded ten-minute limit.')
            if part.stat().st_size != EXPECTED_BYTES or sha256(part) != EXPECTED_SHA:
                raise ValueError('Downloaded size/SHA256 does not match the inspected mirror LFS object.')
            used_url = url
            attempts.append({'url': url, 'status': 'size_and_sha256_verified'})
            break
        except Exception as error:
            message = f'{type(error).__name__}: {error}'
            attempts.append({'url': url, 'status': 'failed', 'error': message})
            print(message, flush=True)
    if used_url is None:
        raise RuntimeError('No bound public distribution downloaded successfully.')
    part.rename(TARGET)
    record = {'dataset': 'PEMS-BAY', 'acquired_at_utc': datetime.now(timezone.utc).isoformat(),
              'original_project': 'https://github.com/liyaguang/DCRNN#data-preparation',
              'original_folder_file_id': '1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq',
              'distribution_url': used_url, 'sha256': EXPECTED_SHA, 'bytes': EXPECTED_BYTES,
              'checksum_source': 'inspected HuggingFace LFS metadata, not an author-published checksum',
              'mirror_repository': 'jimmygao3218/PEMSBAY', 'mirror_revision': REVISION,
              'attempts': attempts, 'traffic_values_deserialized': False,
              'redistribution_authorized': False,
              'license_note': 'Mirror card labels CC-BY-4.0; original dataset redistribution terms not independently verified.'}
    PROVENANCE.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': 'downloaded_and_byte_verified', 'path': str(TARGET),
                      'bytes': EXPECTED_BYTES, 'sha256': EXPECTED_SHA}, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
