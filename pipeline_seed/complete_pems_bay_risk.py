"""Recorded empty-group score adapter; preserve frozen risk training and weights."""
import argparse
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch
import numpy as np
import train_pems_bay_risk as training
from train_pems_bay_forecaster import read_json, write_json
from freeze_external_validation import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',action='store_true')
    args=parser.parse_args()
    output=training.OUTPUT
    path=output/'empty_group_adapter.json'
    weights={p.name:sha256(p) for p in sorted(output.glob('*.pt'))}
    if path.exists():
        amendment=read_json(path)
        assert amendment['adapter_source_sha256']==sha256(Path(__file__))
        assert amendment['preserved_weight_sha256']==weights
    else:
        assert not args.audit and not (output/'results.json').exists()
        assert len(weights)==11
        write_json(path,{'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
            'reason':'A fitted availability group can be empty within an individual online scenario; legacy score concatenates no batches.',
            'change':'For zero input rows only, return an empty float32 score vector. Other scores use the original function.',
            'training_and_model_selection_changed':False,'evaluation_values_read':False,
            'adapter_source_sha256':sha256(Path(__file__)),'preserved_weight_sha256':weights})
    original=training.score
    def score_empty_safe(model,x,*args,**kwargs):
        return np.empty(0,dtype=np.float32) if len(x)==0 else original(model,x,*args,**kwargs)
    with patch.object(training,'score',score_empty_safe):
        if args.audit:
            from audit_pems_bay_risk import main as audit
            audit()
        else: training.main()
    assert {p.name:sha256(p) for p in sorted(output.glob('*.pt'))}==weights


if __name__=='__main__': main()
