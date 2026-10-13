"""Freeze the complete PEMS-BAY evaluation execution before speed unsealing."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
import hashlib
import h5py
import numpy as np
from calibrate_pems_bay_alerts import verify_risk
from train_pems_bay_forecaster import read_json,write_json
from freeze_external_validation import sha256
from pems_bay_reader import SealedPartitionReader

ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/'outputs/pems-bay-evaluation'
SEAL=OUTPUT/'execution_freeze.json'
CAL=ROOT/'outputs/pems-bay-calibration'


def verify_calibration():
    verify_risk()
    for file in ('calibration_implementation_freeze.json','config.json'):
        for name,digest in read_json(CAL/file)['source_sha256'].items(): assert sha256(ROOT/name)==digest,name
    for name,digest in read_json(CAL/'results.json')['artifact_sha256'].items(): assert sha256(CAL/name)==digest,name
    assert read_json(CAL/'audit.json')['passed']
    assert read_json(CAL/'audit.json')['results_sha256']==sha256(CAL/'results.json')


def verify_execution():
    if not SEAL.exists(): raise PermissionError('Evaluation execution freeze is missing')
    seal=read_json(SEAL)
    if seal['stage']!='complete_evaluation_execution_before_unsealing': raise PermissionError('Wrong execution freeze stage')
    for name,digest in seal['source_sha256'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha256(path)!=digest:
            raise PermissionError(f'Execution dependency changed: {name}')
    return seal


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--verify',action='store_true'); args=parser.parse_args()
    if SEAL.exists():
        seal=verify_execution(); print({'execution_freeze':'verified','files':len(seal['source_sha256']),
                                      'evaluation_speed_values_read_by_verifier':False}); return
    if args.verify: raise FileNotFoundError('No execution freeze')
    verify_calibration(); OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'unseal_started.json').exists(): raise ValueError('Unsealing already started; never create a fresh pre-unseal seal')
    design=read_json(ROOT/'configs/pems_bay_validation_v1.json')
    original=h5py.Dataset.__getitem__
    def no_speed(dataset,key,*a,**kw):
        if dataset.name=='/speed/block0_values': raise PermissionError('Freezing may read metadata only')
        return original(dataset,key,*a,**kw)
    with patch.object(h5py.Dataset,'__getitem__',no_speed):
        reader=SealedPartitionReader(ROOT/design['dataset']['local_path'],design)
    lo,hi=reader.bounds['evaluation']; ts=reader.timestamps[lo:hi]
    dates=np.arange(ts[0].astype('datetime64[D]'),ts[-1].astype('datetime64[D]')+np.timedelta64(1,'D'),dtype='datetime64[D]')
    draws=np.random.default_rng(design['evaluation']['bootstrap_seed']).integers(0,len(dates),size=(2000,len(dates)))
    bootstrap=OUTPUT/'bootstrap_draws.npz'
    if bootstrap.exists():
        with np.load(bootstrap) as f: assert np.array_equal(draws,f['draws']) and np.array_equal(dates,f['dates'])
    else: np.savez_compressed(bootstrap,draws=draws,dates=dates)
    manifest={}
    def add(path,digest=None):
        path=path.resolve(); name=path.relative_to(ROOT).as_posix(); current=sha256(path)
        if digest is not None: assert current==digest,name
        if name in manifest: assert manifest[name]==current,name
        manifest[name]=current
    stages=['outputs/pems-bay-validation-plan/design_freeze.json',
            'outputs/pems-bay-binding/reader_implementation_freeze.json','outputs/pems-bay-binding/binding.json',
            'outputs/pems-bay-forecaster/forecast_implementation_freeze.json','outputs/pems-bay-forecaster/config.json','outputs/pems-bay-forecaster/results.json',
            'outputs/pems-bay-risk/risk_implementation_freeze.json','outputs/pems-bay-risk/config.json','outputs/pems-bay-risk/results.json',
            'outputs/pems-bay-calibration/calibration_implementation_freeze.json','outputs/pems-bay-calibration/config.json','outputs/pems-bay-calibration/results.json',
            'outputs/pems-bay-calibration/calibration_choice_freeze.json']
    for name in stages:
        path=ROOT/name; add(path); content=read_json(path)
        for source,digest in content.get('source_sha256',{}).items(): add(ROOT/source,digest)
        for artifact,digest in content.get('artifact_sha256',{}).items(): add(path.parent/artifact,digest)
    for name in ('freeze_pems_bay_evaluation.py','pems_bay_evaluation_gate.py','evaluate_pems_bay_alerts.py',
                 'audit_pems_bay_evaluation.py','summarize_pems_bay_evaluation.py','docs/pems_bay_evaluation_protocol.zh-CN.md'):
        add(ROOT/name)
    add(bootstrap)
    selection=read_json(CAL/'selection.json')
    settings={m:{g:selection['methods'][m][g]['selected'] for g in ('coarse','fine')} for m in selection['methods']}
    seal={'stage':'complete_evaluation_execution_before_unsealing','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
          'protocol_id':design['protocol_id'],'evaluation_values_read_before_freeze':False,
          'evaluation_bounds':list(reader.bounds['evaluation']),'selected_ids':reader.selected_ids,'selected_columns':reader.columns.tolist(),
          'evaluation_dates':dates.astype(str).tolist(),'selected_settings':settings,
          'bootstrap_draws_raw_sha256':hashlib.sha256(draws.tobytes()).hexdigest(),
          'evaluation_design':design['evaluation'],'source_sha256':dict(sorted(manifest.items()))}
    write_json(SEAL,seal)
    print({'execution_freeze':'created','files':len(manifest),'evaluation_days':len(dates),'evaluation_values_read':False})


if __name__=='__main__': main()
