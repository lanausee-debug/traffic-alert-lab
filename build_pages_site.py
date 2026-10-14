"""Build a static Pages artifact from verified aggregate results only."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import verify_release

ROOT=Path(__file__).resolve().parent


def build():
    verify_release.main()
    source=(ROOT/'demo/index.html').read_text(encoding='utf-8')
    # GitHub Pages serves the HTML at the project root; Markdown evidence links
    # should point to the public repository rather than missing hosted files.
    html=re.sub(r'href="\.\./([^"]+)"',
                lambda m:'href="https://github.com/lanausee-debug/traffic-alert-lab/blob/main/'+m[1]+'"',source)
    data=lambda h: re.search(r'<script id="result-data" type="application/json">(.*?)</script>',h,re.S)[1]
    assert data(source)==data(html)
    assert 'joint primary test did not pass' in html
    assert 'href="../' not in html
    assert not any(s in html for s in ('fetch(', 'XMLHttpRequest', '<script src='))
    out=ROOT/'_site'
    out.mkdir(exist_ok=True)
    blobs={'index.html':html.encode('utf-8'),'.nojekyll':b''}
    if any(p.name not in blobs for p in out.iterdir()):
        raise ValueError('Unexpected file in static artifact directory')
    for name,blob in blobs.items():
        p=out/name
        if p.exists():
            if p.read_bytes()!=blob:
                raise ValueError('Use a fresh artifact directory after a source change')
        else:
            with p.open('xb') as stream:
                stream.write(blob)
    report={'static_files':list(blobs),'index_sha256':hashlib.sha256(blobs['index.html']).hexdigest(),
            'embedded_aggregate_bytes_unchanged':True,'primary_metric_supported':False,
            'raw_data_or_weights_included':False}
    print(json.dumps(report))
    return report


if __name__=='__main__':
    build()
