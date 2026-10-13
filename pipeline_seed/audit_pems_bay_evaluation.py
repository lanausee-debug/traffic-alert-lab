"""Pre-unseal zero-value-access audit and independent held-out verification."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import hashlib
from unittest.mock import patch
import h5py
import numpy as np
import torch
from freeze_pems_bay_evaluation import ROOT,OUTPUT,SEAL,CAL,verify_execution
from pems_bay_evaluation_gate import EvaluationReader,guard_evaluation_reads,validate_request
from evaluate_pems_bay_alerts import paired_difference,compare_all,comparison_specs,METRICS,decision
from audit_pems_bay_calibration import (synthetic_checks,catalog,independently_activated,independent_replay,assert_same)
from calibrate_pems_bay_alerts import frozen_scores,METHODS,GRIDS
from train_pems_bay_forecaster import read_json,write_json,load_model
from freeze_external_validation import sha256
from pems_bay_windows import calendar_features
from pems_bay_alert_replay import layout
from congestion_alert_pilot import SCENARIOS,outage_mask,infer
from congestion_risk_pilot import features


def independent_difference(a,b,draws,primary):
    result={}; at=a.sum(axis=1); bt=b.sum(axis=1)
    for name,columns,denominator in METRICS:
        values=[]; undefined=0
        if np.any(at[:,denominator]<=0) or np.any(bt[:,denominator]<=0): point=None
        else:
            individual=[(sum(at[s,j] for j in columns)/at[s,denominator]-sum(bt[s,j] for j in columns)/bt[s,denominator]) for s in range(3)]
            point=sum(individual)/3
        for sample in draws:
            aa=a[:,sample,:].sum(axis=1); bb=b[:,sample,:].sum(axis=1)
            if np.any(aa[:,denominator]<=0) or np.any(bb[:,denominator]<=0): undefined+=1; continue
            values.append(sum((sum(aa[s,j] for j in columns)/aa[s,denominator]-sum(bb[s,j] for j in columns)/bb[s,denominator]) for s in range(3))/3)
        ci=np.quantile(values,[.025,.975]).tolist() if point is not None and undefined==0 else None
        result[name]={'difference':point,'ci95':ci,'undefined_bootstrap_replicates':undefined,
                      'interval_status':'computed' if ci is not None else 'undefined_denominator'}
        if primary and name=='warning10_recall':
            result[name]['ci975']=np.quantile(values,[.0125,.9875]).tolist() if ci is not None else None
    return result


def bootstrap_checks():
    a=np.zeros((3,5,13),np.float64); b=np.zeros_like(a)
    for s in range(3):
        b[s,:,0]=20+s+np.arange(5); b[s,:,1]=3+s+np.arange(5)
        b[s,:,2]=1; b[s,:,3]=2; b[s,:,4]=np.arange(5)%2; b[s,:,11]=1
    a[:]=b; a[:,:,2]+=1; a[:,:,3]+=1; a[:,:,11]+=1; a[:,:,4]+=1
    draws=np.random.default_rng(8120).integers(0,5,size=(2000,5))
    assert_same(paired_difference(a,b,draws,True),independent_difference(a,b,draws,True))
    identical=paired_difference(a,a,draws,True)
    for row in identical.values(): assert row['difference']==0 and row['ci95']==[0.,0.]
    z=a.copy(); z[:,0,1]=0; zero=np.zeros((2000,5),int)
    out=paired_difference(z,z,zero,True)
    assert out['warning10_recall']['ci975'] is None and out['warning10_recall']['undefined_bootstrap_replicates']==2000
    z[:,:,1]=0
    assert paired_difference(z,z,draws,True)['warning10_recall']['difference'] is None
    return {'calendar_pairing_formula_independent_loop':True,'heterogeneous_scenario_rate_mean':True,
            'identical_pair_zero_interval':True,'zero_denominator_no_smoothing_or_redraw':True}


def request_checks(bounds):
    lo,hi=bounds
    cases=[('evaluation',lo,hi,False,False),('calibration',lo,hi,True,False),
           ('evaluation',lo-1,hi,True,False),('evaluation',lo,hi-1,True,False),
           ('evaluation',lo,hi,True,True),('evaluation',-1,hi,True,False)]
    for args in cases:
        partition,start,stop,enabled,consumed=args
        try: validate_request(partition,start,stop,bounds,enabled,consumed)
        except PermissionError: pass
        else: raise AssertionError('Invalid evaluation request accepted')
    validate_request('evaluation',lo,hi,bounds,True,False)
    return {'sealed_default':True,'only_exact_partition':True,'no_partial_or_crossing_reads':True,'once_per_reader':True}


def preflight():
    seal=verify_execution(); path=OUTPUT/'preflight.json'
    if path.exists():
        prior=read_json(path)
        assert prior['passed'] and prior['execution_freeze_sha256']==sha256(SEAL)
        assert prior['audit_source_sha256']==sha256(Path(__file__))
        print('Existing preflight verified without unsealing'); return
    semantics=synthetic_checks(); bootstrap=bootstrap_checks(); requests=request_checks(seal['evaluation_bounds'])
    design=read_json(ROOT/'configs/pems_bay_validation_v1.json')
    original=h5py.Dataset.__getitem__; attempted=[]
    def no_speed(dataset,key,*args,**kwargs):
        if dataset.name=='/speed/block0_values':
            attempted.append(str(key)); raise PermissionError('Preflight never deserializes evaluation speed')
        return original(dataset,key,*args,**kwargs)
    with patch.object(h5py.Dataset,'__getitem__',no_speed):
        reader=EvaluationReader(ROOT/design['dataset']['local_path'],design)
        assert not reader.access_log and list(reader.bounds['evaluation'])==seal['evaluation_bounds']
        try: reader.read_partition('evaluation')
        except PermissionError: pass
        else: raise AssertionError('Default evaluation read allowed')
        try: EvaluationReader(ROOT/design['dataset']['local_path'],design,permit=True)
        except PermissionError: pass
        else: raise AssertionError('Evaluation read enabled before preflight exists')
    assert attempted==[]
    with np.load(OUTPUT/'bootstrap_draws.npz') as f: draws=f['draws']; dates=f['dates']
    expected=np.random.default_rng(8120).integers(0,len(dates),size=(2000,len(dates)))
    assert np.array_equal(draws,expected) and dates.astype(str).tolist()==seal['evaluation_dates']
    assert hashlib.sha256(draws.tobytes()).hexdigest()==seal['bootstrap_draws_raw_sha256']
    selection=read_json(CAL/'selection.json')
    assert list(selection['methods'])==list(METHODS)
    for method in METHODS:
        for grid in GRIDS:
            row=selection['methods'][method][grid]['selected']
            assert row==seal['selected_settings'][method][grid] and row['feasible']
            assert row['confirmation'] in (1,2) and np.isfinite(row['threshold'])
            assert len(row['by_scenario'])==3
    proof={'passed':True,'audited_at_utc':datetime.now(timezone.utc).isoformat(),
           'stage':'before_first_evaluation_speed_access','execution_freeze_sha256':sha256(SEAL),
           'audit_source_sha256':sha256(Path(__file__)),'evaluation_values_read':False,
           'metadata_speed_read_requests':0,'dependency_sha256_files':len(seal['source_sha256']),
           'event_semantics':semantics,'bootstrap_semantics':bootstrap,'reader_request_semantics':requests,
           'selected_settings_verified':14,'bootstrap_calendar_days':len(dates)}
    write_json(path,proof)
    print({'preflight':'passed','evaluation_values_read':False,'frozen_files':len(seal['source_sha256'])},flush=True)


def post_audit():
    seal=verify_execution(); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    result=read_json(OUTPUT/'results.json')
    assert result['execution_freeze_sha256']==sha256(SEAL)
    for name,digest in result['artifact_sha256'].items(): assert sha256(OUTPUT/name)==digest,name
    design=read_json(ROOT/'configs/pems_bay_validation_v1.json')
    reader=EvaluationReader(ROOT/design['dataset']['local_path'],design,permit=True); trace=[]
    with guard_evaluation_reads(reader,trace):
        for forbidden in ('calibration','risk_train','forecast_train'):
            try: reader.read_partition(forbidden)
            except PermissionError: pass
            else: raise AssertionError('Cross-partition audit read allowed')
        raw64,ts=reader.read_partition('evaluation')
        try: reader.read_partition('evaluation')
        except PermissionError: pass
        else: raise AssertionError('Second raw read accepted by the same reader')
        with h5py.File(reader.path,'r') as f:
            try: f['speed/block0_values'][:]
            except PermissionError: pass
            else: raise AssertionError('Full speed matrix allowed')
    assert len(reader.access_log)==len(trace)==1
    with np.load(OUTPUT/'evaluation.npz') as f: part={k:f[k] for k in f.files}
    expected=np.array([t for t in range(12,len(ts)-2) if np.all(np.diff(ts[t-12:t+3])==np.timedelta64(5,'m'))])
    assert np.array_equal(expected,part['origins']) and np.array_equal(part['timestamps'],ts)
    valid=(np.isfinite(raw64)&(raw64>0)).astype(np.float32); origins=part['origins']
    truth=np.stack([raw64[origins+i]<=30 for i in range(3)]).all(axis=0)
    eligible=np.stack([valid[origins+i]>0 for i in range(3)]).all(axis=0)
    assert np.array_equal(truth,part['truth']) and np.array_equal(eligible,part['eligible'])
    assert read_json(OUTPUT/'preparation.json')['reader']==read_json(OUTPUT/'access_log.json')['preparation_read']==reader.access_log[0]
    assert read_json(OUTPUT/'preparation.json')['runtime_interception']==trace[0]
    with np.load(ROOT/'outputs/pems-bay-binding/forecast_training_statistics.npz') as f:
        stats={k:f[k].copy() for k in ('mean','scale','adjacency')}
    models=[load_model(seed,stats['adjacency'],ROOT/f'outputs/pems-bay-forecaster/model_tcn_seed{seed}.pt') for seed in (42,43,44)]
    groups=read_json(ROOT/'outputs/pems-bay-binding/fault_groups.json')
    raw=raw64.astype(np.float32); z=np.where(valid>0,(raw-stats['mean'])/stats['scale'],0).astype(np.float32)
    context=calendar_features(ts); days=(ts.astype('datetime64[D]')-reader.d0).astype(int)
    assert np.array_equal(part['day_indices'],days[origins])
    positions=np.unique(np.linspace(0,len(origins)-1,16,dtype=int)); delta=0.
    for s in SCENARIOS:
        mask,fault=outage_mask(valid,days,s,groups)
        assert np.array_equal(fault[origins-1],part[f'{s}_fault'])
        pred=infer(models,z,mask,origins[positions],context,stats['mean'],stats['scale'])
        error=float(np.max(np.abs(pred-part[f'{s}_predictions'][:,positions]))); delta=max(delta,error); assert error<=1e-4
        first=features(raw,mask,origins[positions],context,stats['adjacency'],part[f'{s}_predictions'][:,positions])
        hidden=raw.copy(); hidden[mask==0]=10000
        second=features(hidden,mask,origins[positions],context,stats['adjacency'],part[f'{s}_predictions'][:,positions])
        for k,a,b in zip(('state','single','group'),first[:3],second[:3]):
            assert np.array_equal(a,part[f'{s}_{k}'][positions]) and np.array_equal(a,b)
        t=int(origins[positions[len(positions)//2]]); altered=raw.copy(); altered[t:]=10000
        altered_z=np.where(valid>0,(altered-stats['mean'])/stats['scale'],0).astype(np.float32)
        before=infer(models,z,mask,np.array([t]),context,stats['mean'],stats['scale'])
        after=infer(models,altered_z,mask,np.array([t]),context,stats['mean'],stats['scale'])
        assert np.array_equal(before,after)
        a=features(raw,mask,np.array([t]),context,stats['adjacency'],before)
        b=features(altered,mask,np.array([t]),context,stats['adjacency'],after)
        for aa,bb in zip(a[:3],b[:3]): assert np.array_equal(aa,bb)
    scores=frozen_scores(part)
    with np.load(OUTPUT/'scores_evaluation.npz') as f:
        assert set(scores)==set(f.files)
        for k,v in scores.items(): assert np.array_equal(v,f[k]),k
    poisoned=dict(part); poisoned['common']=np.zeros_like(part['common']); poisoned['onset']=~part['onset']
    other=frozen_scores(poisoned)
    for k,v in scores.items(): assert np.array_equal(v,other[k])
    selected=read_json(OUTPUT/'selected_settings.json')
    assert selected['settings']==seal['selected_settings'] and selected['selection_source_sha256']==sha256(CAL/'selection.json')
    structure=catalog(part); records=read_json(OUTPUT/'event_records.json'); independent_daily={}; checked=0
    budgets=read_json(CAL/'budgets.json')['budgets']
    with np.load(OUTPUT/'daily_metrics.npz') as saved:
        for method in METHODS:
            for grid in GRIDS:
                setting=selected['settings'][method][grid]
                for s in SCENARIOS:
                    bits=independently_activated(scores[f'{method}__{s}'],setting['threshold'],setting['confirmation'],part)
                    actual,daily,events=independent_replay(part,bits,structure,s)
                    assert_same(actual,result['evaluation'][method][grid][s])
                    key=f'{method}__{grid}__{s}'; independent_daily[key]=daily
                    assert np.allclose(daily,saved[key],rtol=0,atol=1e-10) and events==records[key]
                    rate=actual['false_per_valid_station_day']
                    expected={'calibration_budget':budgets[s],'evaluation_rate':rate,'exceeded':bool(rate is not None and rate>budgets[s]+1e-12)}
                    assert_same(expected,result['calibration_budget_exceedance'][method][grid][s]); checked+=1
                print(f'Independently audited evaluation: {method} {grid}',flush=True)
        assert set(saved.files)==set(independent_daily)==set(records) and checked==42
    with np.load(OUTPUT/'bootstrap_draws.npz') as f: draws=f['draws']; dates=f['dates']
    assert np.array_equal(dates,layout(part)[2]) and dates.astype(str).tolist()==result['evaluation_dates']==seal['evaluation_dates']
    expected=np.random.default_rng(8120).integers(0,len(dates),size=(2000,len(dates)))
    assert np.array_equal(draws,expected)
    for key,cm,cg,bm,bg,primary in comparison_specs():
        a=np.stack([independent_daily[f'{cm}__{cg}__{s}'] for s in SCENARIOS])
        b=np.stack([independent_daily[f'{bm}__{bg}__{s}'] for s in SCENARIOS])
        actual=independent_difference(a,b,draws,primary)
        assert_same(actual,result['comparisons'][key]['metrics'])
    assert decision(result['comparisons'],result['evaluation'])==result['verdict']
    assert not result['evaluation_threshold_grid_scanned'] and not result['models_retrained']
    verify_execution()
    audit={'passed':True,'audited_at_utc':datetime.now(timezone.utc).isoformat(),
           'results_sha256':sha256(OUTPUT/'results.json'),'execution_freeze_sha256':sha256(SEAL),
           'evaluation_values_read_for_verification':True,'evaluation_thresholds_reselected':False,
           'audited_fixed_scenario_operating_points':42,'audited_comparisons':11,'bootstrap_replicates_per_comparison':2000,
           'forecast_sample_max_difference_mph':delta,'reader':reader.access_log,'runtime_interceptions':trace,
           'checks':{'all_preunseal_dependencies_unchanged':True,'independent_Y_eligibility_and_time_windows':True,
             'forecast_feature_samples_and_all_risk_scores':True,'hidden_future_and_supervision_mask_invariance':True,
             'independent_42_event_replays_daily_counts_and_records':True,'independent_bootstrap_and_verdict':True,
             'budget_exceedance_reported':True,'no_threshold_scan_or_retraining':True}}
    if (OUTPUT/'audit.json').exists():
        prior=read_json(OUTPUT/'audit.json'); audit['audited_at_utc']=prior['audited_at_utc']; assert prior==audit
    else: write_json(OUTPUT/'audit.json',audit)
    print({'post_evaluation_audit':'passed','fixed_operating_points':42,'comparisons':11},flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',action='store_true'); parser.add_argument('--self-test',action='store_true'); args=parser.parse_args()
    if args.self_test:
        print({'events':synthetic_checks(),'bootstrap':bootstrap_checks(),'requests':request_checks((28212,52116))})
    elif args.preflight: preflight()
    else: post_audit()


if __name__=='__main__': main()
