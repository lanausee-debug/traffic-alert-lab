"""Seven frozen methods, full coarse/fine calibration only; evaluation sealed."""
from datetime import datetime, timezone
from pathlib import Path
import contextlib
import time
from unittest.mock import patch
import h5py
import numpy as np
import torch
import train_pems_bay_risk as risk
from train_pems_bay_forecaster import read_json,write_json,load_model
from freeze_external_validation import sha256
from pems_bay_reader import SealedPartitionReader
from pems_bay_alert_replay import alarm,replay,layout,select_key,COLUMNS
from congestion_alert_pilot import SCENARIOS

ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/'outputs/pems-bay-calibration'
RISK=risk.OUTPUT
METHODS=risk.METHODS
GRIDS=('coarse','fine')


def verify_risk():
    risk.verify_forecaster()
    for file in ('risk_implementation_freeze.json','config.json'):
        for name,digest in read_json(RISK/file)['source_sha256'].items():
            assert sha256(ROOT/name)==digest,name
    for name,digest in read_json(RISK/'results.json')['artifact_sha256'].items():
        assert sha256(RISK/name)==digest,name
    assert read_json(RISK/'audit.json')['passed']
    for name,digest in read_json(RISK/'empty_group_adapter.json')['preserved_weight_sha256'].items():
        assert sha256(RISK/name)==digest,name


@contextlib.contextmanager
def guard_calibration_reads(reader, trace):
    original=h5py.Dataset.__getitem__; allowed=reader.bounds['calibration']
    columns=sorted(reader.columns.tolist())
    def guarded(dataset,key,*args,**kwargs):
        if dataset.name=='/speed/block0_values':
            if not isinstance(key,tuple) or len(key)!=2 or not isinstance(key[0],slice):
                raise PermissionError('Calibration may not read a full/unbounded speed dataset')
            rows=key[0]
            if (rows.start,rows.stop)!=allowed or rows.step not in (None,1) or np.asarray(key[1]).tolist()!=columns:
                raise PermissionError('Only full named calibration selected-station slice allowed')
            trace.append({'partition':'calibration','row_start':rows.start,'row_stop_exclusive':rows.stop,'columns':columns})
        return original(dataset,key,*args,**kwargs)
    with patch.object(h5py.Dataset,'__getitem__',guarded): yield


def frozen_scores(part):
    logs=read_json(RISK/'model_selection.json')
    fitted={'global_calibration':[risk.load_risk(RISK/'global_calibration.pt')], 'group_calibration':{}}
    for g,rec in logs['group_calibration'].items():
        if 'fallback' not in rec: fitted['group_calibration'][g]=[risk.load_risk(RISK/f'group{g}.pt')]
    for method in ('state_logistic','onset_linear'): fitted[method]=[risk.load_risk(RISK/f'{method}.pt')]
    for method in ('persistent_mlp','onset_mlp'):
        fitted[method]=[risk.load_risk(RISK/f'{method}_seed{seed}.pt') for seed in risk.SEEDS]
    with np.load(RISK/'normalization.npz') as f: norm={k:f[k].copy() for k in f.files}
    original=risk.score
    def empty_safe(model,x,*args,**kwargs):
        return np.empty(0,np.float32) if len(x)==0 else original(model,x,*args,**kwargs)
    with patch.object(risk,'score',empty_safe): return risk.scores(part,fitted,norm)


def thresholds(method,grid,design):
    c=design['calibration']
    if method=='dense_speed':
        step=c[f'speed_{grid}_mph']['step']
        return [-float(v) for v in np.arange(15,60.1,step)]+[c['speed_off_score_threshold']]
    if grid=='coarse': return c['probability_coarse']
    return [i/1000 for i in range(10,901,5)]+[c['probability_fine']['off']]


def operation_key(method,grid,index,confirmation,scenario):
    return f'{method}__{grid}__{index:03d}__c{confirmation}__{scenario}'


def main():
    verify_risk()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    sources=['calibrate_pems_bay_alerts.py','audit_pems_bay_calibration.py','pems_bay_alert_replay.py',
             'docs/pems_bay_calibration_protocol.zh-CN.md','configs/pems_bay_validation_v1.json',
             'outputs/pems-bay-risk/risk_implementation_freeze.json','outputs/pems-bay-risk/model_selection.json',
             'outputs/pems-bay-risk/normalization.npz','train_pems_bay_risk.py','pems_bay_reader.py',
             'congestion_alert_pilot.py','congestion_risk_pilot.py','congestion_objective_diagnostic.py']
    sources += [p.relative_to(ROOT).as_posix() for p in sorted(RISK.glob('*.pt'))]
    design=read_json(ROOT/'configs/pems_bay_validation_v1.json')
    config={'stage':'calibration_only','created_date':'2026-10-11','methods':list(METHODS),
            'source_sha256':{name:sha256(ROOT/name) for name in sources},
            'calibration':design['calibration'],'daily_columns':COLUMNS,
            'evaluation_values_read':False,'risk_weights_retrained':False,
            'torch_version':torch.__version__,'numpy_version':np.__version__,'device':'cpu'}
    OUTPUT.mkdir(parents=True,exist_ok=True); config_path=OUTPUT/'config.json'
    if config_path.exists(): assert read_json(config_path)==config,'Configuration changed; preserve original run'
    else: write_json(config_path,config)
    if (OUTPUT/'results.json').exists():
        for name,digest in read_json(OUTPUT/'results.json')['artifact_sha256'].items(): assert sha256(OUTPUT/name)==digest,name
        print('Existing completed calibration verified; no rescan'); return
    begun=time.perf_counter()
    reader=SealedPartitionReader(ROOT/design['dataset']['local_path'],design)
    interceptions=[]
    with guard_calibration_reads(reader,interceptions):
        cache_path=OUTPUT/'calibration.npz'; prep_path=OUTPUT/'preparation.json'
        if cache_path.exists() and prep_path.exists():
            info=read_json(prep_path)
            assert info['config_sha256']==sha256(config_path) and info['cache_sha256']==sha256(cache_path)
            with np.load(cache_path) as f: part={k:f[k] for k in f.files}
        elif cache_path.exists() or prep_path.exists(): raise ValueError('Incomplete calibration cache preserved')
        else:
            with np.load(ROOT/'outputs/pems-bay-binding/forecast_training_statistics.npz') as f:
                statistics={k:f[k].copy() for k in ('mean','scale','adjacency')}
            models=[load_model(seed,statistics['adjacency'],risk.FORE/f'model_tcn_seed{seed}.pt') for seed in (42,43,44)]
            groups=read_json(ROOT/'outputs/pems-bay-binding/fault_groups.json')
            part,names,summary=risk.prepare(reader,'calibration',models,statistics,groups)
            np.savez_compressed(cache_path,**part)
            info={'summary':summary,'feature_names':names,'config_sha256':sha256(config_path),
                  'cache_sha256':sha256(cache_path),'reader':reader.access_log[-1],'runtime_interception':interceptions[-1]}
            write_json(prep_path,info)
        scores=frozen_scores(part)
        np.savez_compressed(OUTPUT/'scores_calibration.npz',**scores)
        write_json(OUTPUT/'access_log.json',{'reader_this_invocation':reader.access_log,
                  'runtime_this_invocation':interceptions,'preparation_read':info['reader'],
                  'evaluation_values_read':False})
        budget_rows={}; budgets={}
        for s in SCENARIOS:
            row,_,_=replay(part,alarm(scores[f'dense_speed__{s}'],-30.,1,part),details=True,scenario=s)
            if not row['events'] or not row['exposure']:
                write_json(OUTPUT/'infeasible.json',{'reason':'No complete calibration events or exposure','scenario':s,'row':row})
                raise ValueError('Frozen calibration plan infeasible; do not retune')
            budget_rows[s]=row; budgets[s]=row['false_per_valid_station_day']
        print(f'Calibration budgets: {budgets}',flush=True)
        write_json(OUTPUT/'budgets.json',{'budgets':budgets,'reference_speed_mph':30,'confirmation':1,'rows':budget_rows})
        selections={}; daily_grid={}; selected_daily={}; selected_rows={}; selected_records={}
        scan_started=time.perf_counter()
        for method in METHODS:
            selections[method]={}; selected_rows[method]={}
            for grid in GRIDS:
                rows=[]; candidates=[]
                for index,t in enumerate(thresholds(method,grid,design)):
                    for c in (1,2):
                        by_s={}
                        for s in SCENARIOS:
                            activated=alarm(scores[f'{method}__{s}'],t,c,part)
                            stats,days,_=replay(part,activated,details=True)
                            by_s[s]=stats
                            daily_grid[operation_key(method,grid,index,c,s)]=days
                        feasible=all(by_s[s]['false_per_valid_station_day']<=budgets[s]+1e-12 for s in SCENARIOS)
                        rec={'grid_index':index,'threshold':t,'confirmation':c,'feasible':feasible,'by_scenario':by_s}
                        rows.append(rec)
                        if feasible: candidates.append((select_key(list(by_s.values()),c,index),rec))
                assert candidates
                chosen=min(candidates,key=lambda v:v[0])[1]
                selections[method][grid]={'selected':chosen,'grid':rows}
                selected_rows[method][grid]={}
                for s in SCENARIOS:
                    activated=alarm(scores[f'{method}__{s}'],chosen['threshold'],chosen['confirmation'],part)
                    stats,days,records=replay(part,activated,details=True,scenario=s)
                    selected_rows[method][grid][s]=stats
                    key=f'{method}__{grid}__{s}'
                    selected_daily[key]=days; selected_records[key]=records
                    assert np.array_equal(days,daily_grid[operation_key(method,grid,chosen['grid_index'],chosen['confirmation'],s)])
                print(f'{method} {grid}: threshold={chosen["threshold"]:g}, c={chosen["confirmation"]}, elapsed={time.perf_counter()-scan_started:.1f}s',flush=True)
            fine_map={(r['threshold'],r['confirmation']):r for r in selections[method]['fine']['grid']}
            for row in selections[method]['coarse']['grid']:
                fine=fine_map[(row['threshold'],row['confirmation'])]
                assert row['by_scenario']==fine['by_scenario'] and row['feasible']==fine['feasible']
            a=selections[method]['fine']['selected']; b=selections[method]['coarse']['selected']
            assert select_key(list(a['by_scenario'].values()),a['confirmation'],0)[:3]<=select_key(list(b['by_scenario'].values()),b['confirmation'],0)[:3]
        write_json(OUTPUT/'selection.json',{'budgets':budgets,'methods':selections,
                   'rank_minimize':design['calibration']['rank_minimize'],'selected_before_evaluation_values_read':True})
        np.savez_compressed(OUTPUT/'calibration_grid_daily.npz',**daily_grid)
        np.savez_compressed(OUTPUT/'calibration_selected_daily.npz',**selected_daily)
        write_json(OUTPUT/'event_records.json',selected_records)
        write_json(OUTPUT/'calibration_choice_freeze.json',{'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
                  'evaluation_values_read':False,'artifact_sha256':{name:sha256(OUTPUT/name) for name in
                    ('selection.json','budgets.json','scores_calibration.npz','calibration_grid_daily.npz','calibration_selected_daily.npz','event_records.json')}})
    for name,digest in config['source_sha256'].items(): assert sha256(ROOT/name)==digest,name
    artifacts=sorted(p.name for p in OUTPUT.iterdir() if p.is_file() and p.name!='results.json')
    result={'completed_at_utc':datetime.now(timezone.utc).isoformat(),'stage':'calibration_only',
            'evaluation_values_read':False,'held_out_evaluation_executed':False,'risk_weights_retrained':False,
            'complete_calibration_events':budget_rows[SCENARIOS[0]]['events'],
            'valid_station_days':budget_rows[SCENARIOS[0]]['exposure'],
            'calibration_dates':layout(part)[2].astype(str).tolist(),
            'calibration_segments':len(layout(part)[0]),'budgets':budgets,'selected_service_metrics':selected_rows,
            'enumerated_operating_points':sum(len(v[g]['grid']) for v in selections.values() for g in GRIDS),
            'scenario_operating_records':len(daily_grid),'threshold_scan_seconds':time.perf_counter()-scan_started,
            'elapsed_seconds':time.perf_counter()-begun,'artifact_sha256':{name:sha256(OUTPUT/name) for name in artifacts}}
    write_json(OUTPUT/'results.json',result)
    print({'status':'calibration_completed','events':result['complete_calibration_events'],
           'operating_points':result['enumerated_operating_points'],'seconds':result['elapsed_seconds'],
           'evaluation_values_read':False},flush=True)


if __name__=='__main__': main()
