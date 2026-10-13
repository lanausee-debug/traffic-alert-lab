"""Independent native-clock alert matching and full calibration-grid audit."""
import argparse
from datetime import datetime,timezone
import time
import h5py
import numpy as np
import torch
from calibrate_pems_bay_alerts import (ROOT,OUTPUT,RISK,METHODS,GRIDS,verify_risk,guard_calibration_reads,
                                      frozen_scores,thresholds,operation_key)
from train_pems_bay_forecaster import read_json,write_json,load_model
from freeze_external_validation import sha256
from pems_bay_reader import SealedPartitionReader
from pems_bay_windows import calendar_features
from pems_bay_alert_replay import alarm,replay
from congestion_alert_pilot import SCENARIOS,infer,outage_mask,event_stats
from congestion_risk_pilot import features


def intervals(bits):
    bits=np.asarray(bits,dtype=bool)
    starts=np.flatnonzero(bits & ~np.r_[False,bits[:-1]])
    ends=np.flatnonzero(bits & ~np.r_[bits[1:],False])+1
    return list(zip(starts.tolist(),ends.tolist()))


def catalog(part):
    times=part['timestamps'][part['origins']]
    boundaries=[0]+[i for i in range(1,len(times))
        if times[i]-times[i-1]!=np.timedelta64(5,'m') or part['origins'][i]-part['origins'][i-1]!=1]+[len(times)]
    d0=part['timestamps'][0].astype('datetime64[D]')
    n_days=int((part['timestamps'][-1].astype('datetime64[D]')-d0)/np.timedelta64(1,'D'))+1
    day_ids=((times.astype('datetime64[D]')-d0)/np.timedelta64(1,'D')).astype(int)
    blocks=[]; daily=np.zeros((n_days,13),np.float64)
    for node in range(part['truth'].shape[1]):
        daily[:,0]+=np.bincount(day_ids,weights=part['eligible'][:,node].astype(float)/288.,minlength=n_days)
    for first,last in zip(boundaries[:-1],boundaries[1:]):
        for node in range(part['truth'].shape[1]):
            for a,b in intervals(part['eligible'][first:last,node]):
                lo,hi=first+a,first+b; complete=[]; censored=[]
                for c,d in intervals(part['truth'][lo:hi,node]):
                    s,e=lo+c,lo+d
                    if s-lo<2 or e==hi:
                        censored.append((s,e)); daily[day_ids[s],7]+=1
                    else:
                        complete.append((s,e)); daily[day_ids[s],1]+=1
                blocks.append((first,last,node,lo,hi,complete,censored))
    return blocks,day_ids,daily


def independently_activated(scores,t,c,part):
    result=scores>=t
    if c==2:
        times=part['timestamps'][part['origins']]
        out=np.zeros_like(result)
        for i in range(1,len(times)):
            if times[i]-times[i-1]==np.timedelta64(5,'m') and part['origins'][i]-part['origins'][i-1]==1:
                out[i]=result[i]&result[i-1]
        return out
    return result


def independent_points(truth,eligible,activated):
    tp=int(np.count_nonzero(truth&eligible&activated)); fp=int(np.count_nonzero(~truth&eligible&activated))
    fn=int(np.count_nonzero(truth&eligible&~activated)); negatives=int(np.count_nonzero(~truth&eligible))
    return {'valid_points':int(eligible.sum()),'positive_points':int((truth&eligible).sum()),
            'true_positives':tp,'false_positives':fp,'false_negatives':fn,
            'recall':tp/(tp+fn) if tp+fn else None,'precision':tp/(tp+fp) if tp+fp else None,
            'false_positive_rate':fp/negatives if negatives else None}


def independent_replay(part,bits,structure,scenario=None):
    blocks,day_ids,base=structure; daily=base.copy(); records=[]; leads=[]; delays=[]
    for first,last,node,lo,hi,complete,censored in blocks:
        recs=[]
        for s,e in complete:
            rec={'node':node,'event_start_origin':int(part['origins'][s]),
                 'timestamp':str(part['timestamps'][part['origins'][s]]),
                 'event_end_origin_exclusive':int(part['origins'][e-1]+1),'proxy_positive_steps':e-s,
                 'status':'missed','alarm_start_origin':None,'lead_minutes':None,'late_minutes':None,
                 'alarm_start_point_label':None,'any_alarm_match_window':bool(bits[s-2:e,node].any()),
                 'any_alarm_warning_window':bool(bits[s-2:s+1,node].any()),
                 'active_spanning_warning_window_entry':bool(s-3>=lo and bits[s-3,node] and bits[s-2,node]),
                 'segment_first_origin':int(part['origins'][first]),'segment_last_origin':int(part['origins'][last-1])}
            recs.append(rec)
        claimed=set()
        for a,b in intervals(bits[lo:hi,node]):
            a,b=lo+a,lo+b; day=day_ids[a]
            if a==lo or b==hi: daily[day,8]+=1; continue
            if any(s-2<=a<e for s,e in censored): daily[day,9]+=1; continue
            choices=[j for j,(s,e) in enumerate(complete) if s-2<=a<e]
            unused=[j for j in choices if j not in claimed]
            if unused:
                j=unused[0]; claimed.add(j); s,e=complete[j]; rec=recs[j]
                rec['alarm_start_origin']=int(part['origins'][a]); rec['alarm_start_point_label']=bool(part['truth'][a,node])
                daily[day_ids[s],3]+=1
                if a<=s:
                    lead=(s-a+1)*5; daily[day_ids[s],2]+=1; daily[day_ids[s],9+lead//5]+=1
                    rec['status']='warned'; rec['lead_minutes']=lead; leads.append(lead)
                else:
                    late=(a-s-1)*5; daily[day_ids[s],6]+=1
                    rec['status']='late'; rec['late_minutes']=late; delays.append(late)
            elif choices: daily[day,5]+=1
            else: daily[day,4]+=1
        records.extend(recs)
    total=daily.sum(axis=0); events=int(total[1]); exposure=total[0]
    columns=('exposure','events','warned','detected','false','duplicate','late','censored_events','censored_alerts','unknown_alerts')
    row={k:float(v) for k,v in zip(columns,total[:10])}
    row.update({'warning_recall':float(total[2]/events) if events else None,
                'detection_recall':float(total[3]/events) if events else None,
                'false_per_valid_station_day':float(total[4]/exposure) if exposure else None,
                'missed_events':int(events-total[3]),
                'warning_recall_at_least10':float((total[11]+total[12])/events) if events else None,
                'warning_lead_counts':{str(k):int(total[9+k//5]) for k in (5,10,15)},
                'warning_lead_mean_minutes':float(np.mean(leads)) if leads else None,
                'warning_lead_median_minutes':float(np.median(leads)) if leads else None,
                'late_detection_mean_minutes':float(np.mean(delays)) if delays else None,
                'point_all':independent_points(part['truth'],part['eligible'],bits)})
    if scenario is not None:
        fault=part[f'{scenario}_fault']
        row['point_fault_nodes']=independent_points(part['truth'],part['eligible']&fault,bits)
        row['point_other_nodes']=independent_points(part['truth'],part['eligible']&~fault,bits)
    return row,daily,records


def assert_same(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b),(set(a),set(b))
        for k in a: assert_same(a[k],b[k])
    elif isinstance(a,float):
        assert b is not None and np.isclose(a,b,rtol=1e-12,atol=1e-10),(a,b)
    else: assert a==b,(a,b)


def synthetic_checks():
    def part_of(truth,eligible=None):
        n=len(truth)
        return {'origins':np.arange(n), 'timestamps':np.datetime64('2017-01-01T23:00')+np.arange(n)*np.timedelta64(5,'m'),
                'truth':truth,'eligible':np.ones_like(truth) if eligible is None else eligible,
                'common':np.zeros_like(truth),'onset':np.zeros_like(truth)}
    truth=np.zeros((40,1),bool); truth[6:10]=True; truth[18:24]=True
    scores=np.zeros((40,1),np.float32)
    scores[4:6]=.5; scores[8:9]=.5; scores[13:15]=.5; scores[19:21]=.5; scores[30:32]=.5
    part=part_of(truth)
    for c in (1,2):
        bits=alarm(scores,.2,c,part); independent=independently_activated(scores,.2,c,part)
        assert np.array_equal(bits,independent)
        a,ad,ar=replay(part,bits,True); b,bd,br=independent_replay(part,bits,catalog(part))
        assert_same(a,b); assert np.allclose(ad,bd,rtol=0,atol=1e-12); assert ar==br
        original,days,_=event_stats(truth,part['eligible'],bits,np.r_[np.zeros(12,int),np.ones(28,int)])
        assert_same({k:a[k] for k in original},original); assert np.array_equal(ad[:,:10],days)
        assert a['events']==2 and a['false']>0
        if c==1: assert a['duplicate']>0
    gap=part_of(truth); gap['timestamps'][15:]+=np.timedelta64(65,'m')
    on=np.zeros_like(scores); on[14:17]=1
    bits=alarm(on,.5,2,gap)
    assert not bits[15,0] and bits[16,0]
    a,ad,ar=replay(gap,bits,True); b,bd,br=independent_replay(gap,bits,catalog(gap))
    assert_same(a,b); assert np.allclose(ad,bd); assert ar==br
    # Consecutive timestamps over midnight do not reset confirmation.
    on=np.zeros_like(scores); on[11:14]=1
    assert alarm(on,.5,2,part)[12,0]
    invalid=part_of(truth); invalid['eligible'][5:8]=False
    bits=alarm(scores,.2,1,invalid)
    a,ad,ar=replay(invalid,bits,True); b,bd,br=independent_replay(invalid,bits,catalog(invalid))
    assert_same(a,b); assert np.allclose(ad,bd); assert ar==br
    noevents=part_of(np.zeros((40,1),bool)); bits=np.zeros((40,1),bool); bits[10:12]=True
    a,ad,ar=replay(noevents,bits,True); b,bd,br=independent_replay(noevents,bits,catalog(noevents))
    assert_same(a,b); assert ar==br==[] and a['events']==0 and a['false']==1
    # Higher threshold can split a single false alarm into two.
    low=np.zeros((40,1),np.float32); low[4:7,0]=[.3,.15,.3]
    assert replay(noevents,alarm(low,.1,1,noevents))[0]['false']==1
    assert replay(noevents,alarm(low,.2,1,noevents))[0]['false']==2
    # Frozen threshold direction: negative mph is larger when traffic is slower.
    speed=np.full((40,1),-40.,np.float32); speed[7:9]=-25
    assert np.flatnonzero(alarm(speed,-30,1,part)[:,0]).tolist()==[7,8]
    return {'continuous_legacy_equality':True,'gap_confirmation_reset_and_segment_scoring':True,
            'midnight_continuity':True,'unknown_truth_and_censoring':True,'empty_complete_event_segment':True,
            'duplicate_and_late_matching':True,'nonmonotonic_false_alert_counts':True,'negative_mph_direction':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--self-test',action='store_true'); args=parser.parse_args()
    semantic=synthetic_checks()
    if args.self_test: print(semantic); return
    verify_risk(); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    config,result=read_json(OUTPUT/'config.json'),read_json(OUTPUT/'results.json')
    for name,digest in config['source_sha256'].items(): assert sha256(ROOT/name)==digest,name
    for name,digest in result['artifact_sha256'].items(): assert sha256(OUTPUT/name)==digest,name
    choice=read_json(OUTPUT/'calibration_choice_freeze.json')
    for name,digest in choice['artifact_sha256'].items(): assert sha256(OUTPUT/name)==digest,name
    design=read_json(ROOT/'configs/pems_bay_validation_v1.json')
    reader=SealedPartitionReader(ROOT/design['dataset']['local_path'],design); interceptions=[]
    with np.load(OUTPUT/'calibration.npz') as f: part={k:f[k] for k in f.files}
    with guard_calibration_reads(reader,interceptions):
        for name in ('evaluation','risk_train','forecast_validation'):
            try: reader.read_partition(name)
            except PermissionError: pass
            else: raise AssertionError('Non-calibration speed read allowed')
        with h5py.File(reader.path,'r') as f:
            try: f['speed/block0_values'][:]
            except PermissionError: pass
            else: raise AssertionError('Unbounded speed read allowed')
        raw64,ts=reader.read_partition('calibration')
    assert len(reader.access_log)==len(interceptions)==1
    expected=np.array([t for t in range(12,len(ts)-2) if np.all(np.diff(ts[t-12:t+3])==np.timedelta64(5,'m'))])
    assert np.array_equal(expected,part['origins']) and np.array_equal(ts,part['timestamps'])
    valid=(np.isfinite(raw64)&(raw64>0)).astype(np.float32); origins=part['origins']
    truth=np.stack([raw64[origins+i]<=30 for i in range(3)]).all(axis=0)
    eligible=np.stack([valid[origins+i]>0 for i in range(3)]).all(axis=0)
    assert np.array_equal(truth,part['truth']) and np.array_equal(eligible,part['eligible'])
    prep=read_json(OUTPUT/'preparation.json'); access=read_json(OUTPUT/'access_log.json')
    assert prep['reader']==access['preparation_read']==reader.access_log[0]
    assert prep['runtime_interception']==interceptions[0]
    with np.load(ROOT/'outputs/pems-bay-binding/forecast_training_statistics.npz') as f:
        stats={k:f[k].copy() for k in ('mean','scale','adjacency')}
    models=[load_model(seed,stats['adjacency'],ROOT/f'outputs/pems-bay-forecaster/model_tcn_seed{seed}.pt') for seed in (42,43,44)]
    groups=read_json(ROOT/'outputs/pems-bay-binding/fault_groups.json')
    raw=raw64.astype(np.float32); z=np.where(valid>0,(raw-stats['mean'])/stats['scale'],0).astype(np.float32)
    context=calendar_features(ts); days=(ts.astype('datetime64[D]')-reader.d0).astype(int)
    assert np.array_equal(part['day_indices'],days[origins])
    positions=np.unique(np.linspace(0,len(origins)-1,12,dtype=int)); delta=0.
    for s in SCENARIOS:
        mask,fault=outage_mask(valid,days,s,groups)
        assert np.array_equal(fault[origins-1],part[f'{s}_fault'])
        prediction=infer(models,z,mask,origins[positions],context,stats['mean'],stats['scale'])
        error=float(np.max(np.abs(prediction-part[f'{s}_predictions'][:,positions]))); delta=max(delta,error); assert error<=1e-4
        feature=features(raw,mask,origins[positions],context,stats['adjacency'],part[f'{s}_predictions'][:,positions])
        hidden=raw.copy(); hidden[mask==0]=10000
        altered=features(hidden,mask,origins[positions],context,stats['adjacency'],part[f'{s}_predictions'][:,positions])
        for k,a,b in zip(('state','single','group'),feature[:3],altered[:3]):
            assert np.array_equal(a,part[f'{s}_{k}'][positions]) and np.array_equal(a,b)
    scores=frozen_scores(part)
    with np.load(OUTPUT/'scores_calibration.npz') as f:
        assert set(scores)==set(f.files)
        for k,v in scores.items(): assert np.array_equal(v,f[k]),k
    # Supervision six-reading mask never enters online scoring or event denominators.
    poisoned=dict(part); poisoned['common']=np.zeros_like(part['common']); poisoned['onset']=~part['onset']
    other=frozen_scores(poisoned)
    for k,v in scores.items(): assert np.array_equal(v,other[k])
    structure=catalog(part); budgets=read_json(OUTPUT/'budgets.json'); selections=read_json(OUTPUT/'selection.json')
    for s in SCENARIOS:
        bits=independently_activated(scores[f'dense_speed__{s}'],-30.,1,part)
        row,_,_=independent_replay(part,bits,structure,s)
        assert_same(row,budgets['rows'][s]); assert np.isclose(row['false_per_valid_station_day'],budgets['budgets'][s],rtol=0,atol=1e-12)
    assert budgets['budgets']==selections['budgets']==result['budgets']
    records=read_json(OUTPUT/'event_records.json'); audited=0; started=time.perf_counter()
    with np.load(OUTPUT/'calibration_grid_daily.npz') as all_days, np.load(OUTPUT/'calibration_selected_daily.npz') as selected_days:
        expected_keys=set(); expected_selected=set()
        for method in METHODS:
            for grid in GRIDS:
                declared=thresholds(method,grid,design); saved=selections['methods'][method][grid]; rows=saved['grid']
                assert len(rows)==len(declared)*2; candidates=[]
                for i,row in enumerate(rows):
                    index,c=divmod(i,2); c+=1
                    assert (row['grid_index'],row['threshold'],row['confirmation'])==(index,declared[index],c)
                    stats=[]
                    for s in SCENARIOS:
                        bits=independently_activated(scores[f'{method}__{s}'],row['threshold'],c,part)
                        actual,days,_=independent_replay(part,bits,structure)
                        assert_same(actual,row['by_scenario'][s]); stats.append(actual)
                        key=operation_key(method,grid,index,c,s); expected_keys.add(key)
                        assert np.allclose(days,all_days[key],rtol=0,atol=1e-10),key
                        audited+=1
                    feasible=all(stats[j]['false_per_valid_station_day']<=budgets['budgets'][s]+1e-12 for j,s in enumerate(SCENARIOS))
                    assert feasible==row['feasible']
                    if feasible:
                        key=(-sum(r['warning_recall'] for r in stats)/3,-min(r['warning_recall'] for r in stats),
                             sum(r['false_per_valid_station_day'] for r in stats)/3,c,index)
                        candidates.append((key,row))
                assert min(candidates,key=lambda v:v[0])[1]==saved['selected']
                chosen=saved['selected']
                for s in SCENARIOS:
                    bits=independently_activated(scores[f'{method}__{s}'],chosen['threshold'],chosen['confirmation'],part)
                    actual,days,events=independent_replay(part,bits,structure,s)
                    assert_same(actual,result['selected_service_metrics'][method][grid][s])
                    key=f'{method}__{grid}__{s}'; expected_selected.add(key)
                    assert np.allclose(days,selected_days[key],rtol=0,atol=1e-10) and events==records[key]
                print(f'Audited {method} {grid}, records={audited}, seconds={time.perf_counter()-started:.1f}',flush=True)
            fine={(r['threshold'],r['confirmation']):r for r in selections['methods'][method]['fine']['grid']}
            for row in selections['methods'][method]['coarse']['grid']:
                assert_same(row['by_scenario'],fine[(row['threshold'],row['confirmation'])]['by_scenario'])
        assert expected_keys==set(all_days.files) and expected_selected==set(selected_days.files)==set(records)
    assert audited==result['scenario_operating_records']==7764 and result['enumerated_operating_points']==2588
    audit={'passed':True,'audited_at_utc':datetime.now(timezone.utc).isoformat(),'stage':'calibration_only',
           'evaluation_values_read':False,'results_sha256':sha256(OUTPUT/'results.json'),
           'audited_scenario_operating_records':audited,'forecast_sample_max_difference_mph':delta,
           'synthetic_checks':semantic,'checks':{'source_and_artifact_sha256':True,'original_three_reading_Y_eligibility':True,
             'frozen_predictor_feature_and_risk_score_reconstruction':True,'supervision_mask_not_online_filter':True,
             'independent_full_grid_matching_counts_exposure_and_leads':True,'budget_and_feasibility_and_selection':True,
             'coarse_grid_subset_and_exact_shared_values':True,'selected_event_records':True,'calibration_only_reads':True},
           'reader':reader.access_log,'runtime_interceptions':interceptions}
    if (OUTPUT/'audit.json').exists():
        prior=read_json(OUTPUT/'audit.json'); audit['audited_at_utc']=prior['audited_at_utc']; assert prior==audit
    else: write_json(OUTPUT/'audit.json',audit)
    print({'audit':'passed','audited_scenario_operating_records':audited,'evaluation_values_read':False},flush=True)


if __name__=='__main__': main()
