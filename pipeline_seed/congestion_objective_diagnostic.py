"""Frozen-model diagnosis of point BCE versus event-start warning evaluation."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch

from recovery_pilot import ROOT, write_json
from multifault_pilot import sha256
from congestion_alert_pilot import SCENARIOS, runs, labels, event_stats
from congestion_risk_pilot import sigmoid, score, online_alarm, pool
from congestion_ablation import OLD, SEEDS, load_data, load_parts, load_model

OUTPUT = ROOT/'outputs/congestion-objective-diagnostic'
PHASES = ('other_negative', 'pre15_negative', 'pre10_negative', 'onset_positive',
          'interior_positive', 'censored_positive')
PARTITIONS = [('training',42,70,3), ('validation',70,77,1),
              ('calibration',77,91,1), ('evaluation',91,119,1)]
METHODS = ('dense_speed', 'state_logistic', 'risk_mlp')


def phase_map(truth, eligible):
    phase = np.full(truth.shape, -1, np.int8)
    phase[eligible & ~truth] = 0
    phase[eligible & truth] = 5
    events = []
    for node in range(truth.shape[1]):
        for lo,hi in runs(eligible[:,node]):
            for a,b in runs(truth[lo:hi,node]):
                s,e = int(lo+a),int(lo+b)
                if s-2<lo or e==hi:
                    continue
                assert not truth[s-2:s,node].any()
                assert np.all(phase[s-2:s,node]==0)
                phase[s-2,node],phase[s-1,node],phase[s,node] = 1,2,3
                phase[s+1:e,node] = 4
                events.append({'node':node,'s':s,'e':e,'lo':int(lo),'hi':int(hi)})
    return phase,events


def build_maps(parts, data):
    raw,_,valid,ts,_,_,_,_,day,_ = data
    maps,summary = {},{}
    for name,a,b,stride in PARTITIONS:
        origins = np.arange(np.searchsorted(day,a)+12,np.searchsorted(day,b)-2)
        window = origins[:,None]+np.arange(-12,3)[None]
        regular = (np.diff(ts[window],axis=1)==np.timedelta64(5,'m')).all(axis=1)
        truth,eligible,_ = labels(raw,valid,origins,regular)
        phase,events = phase_map(truth,eligible)
        selected = parts[name]['natural']['origins']-origins[0]
        assert np.array_equal(origins[selected],parts[name]['natural']['origins'])
        assert np.array_equal(truth[selected],parts[name]['natural']['truth'])
        assert np.array_equal(eligible[selected],parts[name]['natural']['eligible'])
        samplemask = np.zeros(len(origins),bool); samplemask[selected]=True
        durations = np.array([r['e']-r['s'] for r in events])
        counts = lambda m: {p:int((m==i).sum()) for i,p in enumerate(PHASES)}
        sampled_counts = counts(phase[selected])
        summary[name] = {'stride_minutes':5*stride,'complete_events_dense':len(events),
                         'dense_unique_phase_counts':counts(phase),
                         'sampled_unique_phase_counts':sampled_counts,
                         'pooled_training_or_scoring_phase_counts':{k:v*len(SCENARIOS) for k,v in sampled_counts.items()},
                         'events_with_sampled_positive':sum(bool(samplemask[r['s']:r['e']].any()) for r in events),
                         'events_with_sampled_onset':sum(bool(samplemask[r['s']]) for r in events),
                         'events_with_sampled_pre15_or_pre10':sum(bool(samplemask[r['s']-2:r['s']].any()) for r in events),
                         'events_with_sampled_warning_window':sum(bool(samplemask[r['s']-2:r['s']+1].any()) for r in events),
                         'proxy_event_positive_steps_mean':float(durations.mean()),
                         'proxy_event_positive_steps_median':float(np.median(durations)),
                         'proxy_event_positive_steps_quantiles':np.quantile(durations,[.25,.5,.75,.9]).tolist(),
                         'invalid_dense_points':int((phase<0).sum()),
                         'invalid_sampled_points':int((phase[selected]<0).sum())}
        maps[name] = {'phase':phase,'events':events,'selected':selected,'origins':origins,
                      'truth':truth,'eligible':eligible}
    return maps,summary


def rebuild_logits(parts, mean, std):
    linear,record = load_model(OLD/'logistic.pt')
    mlps = [load_model(OLD/f'mlp{seed}.pt')[0] for seed in SEEDS]
    all_logits = {}
    for partition,rows in parts.items():
        all_logits[partition] = {}
        for s,p in rows.items():
            x = ((p['state'].reshape(-1,51)-mean)/std).astype(np.float32)
            all_logits[partition][f'state_logistic_{s}'] = score(linear,x).reshape(p['truth'].shape)
            all_logits[partition][f'risk_mlp_{s}'] = np.stack([score(m,x).reshape(p['truth'].shape) for m in mlps])
    return all_logits


def loss_rows(logits, truth, phase):
    # Average per-seed loss/local gradient, rather than loss of averaged probability.
    if logits.ndim==2:
        logits = logits[None]
    y = truth.astype(np.float32)
    loss = np.mean(np.logaddexp(0,logits)-logits*y[None],axis=0,dtype=np.float64)
    gradient = np.mean(np.abs(sigmoid(logits)-y[None]),axis=0,dtype=np.float64)
    valid = phase>=0; positive = valid & truth
    total_loss = float(loss[valid].sum()); total_gradient = float(gradient[valid].sum())
    pos_loss = float(loss[positive].sum()); pos_gradient = float(gradient[positive].sum())
    pos_count = int(positive.sum())
    rows = {}
    for i,label in enumerate(PHASES):
        chosen = phase==i; count=int(chosen.sum())
        ls = float(loss[chosen].sum()); gs=float(gradient[chosen].sum())
        rows[label] = {'count':count,'sample_share_all':count/int(valid.sum()),
                       'loss_sum':ls,'mean_bce':ls/count if count else None,
                       'loss_share_all':ls/total_loss,'local_gradient_sum':gs,
                       'mean_abs_logit_gradient':gs/count if count else None,
                       'local_gradient_share_all':gs/total_gradient,
                       'positive_sample_share':count/pos_count if i>=3 else None,
                       'positive_loss_share':ls/pos_loss if i>=3 else None,
                       'positive_local_gradient_share':gs/pos_gradient if i>=3 else None}
    return {'valid_points':int(valid.sum()),'positive_points':pos_count,
            'mean_bce':total_loss/int(valid.sum()),'loss_sum':total_loss,
            'positive_loss_sum':pos_loss,'positive_loss_share_all':pos_loss/total_loss,
            'local_gradient_sum':total_gradient,'positive_local_gradient_sum':pos_gradient,
            'phases':rows}


def loss_diagnostic(all_logits,parts,maps):
    result = {}
    for partition,bykey in all_logits.items():
        result[partition] = {}
        phase = maps[partition]['phase'][maps[partition]['selected']]
        for method in ('state_logistic','risk_mlp'):
            logits = np.concatenate([bykey[f'{method}_{s}'][None] if method=='state_logistic' else bykey[f'{method}_{s}'] for s in SCENARIOS],axis=1)
            truth = np.concatenate([parts[partition][s]['truth'] for s in SCENARIOS],axis=0)
            cat = np.concatenate([phase for s in SCENARIOS],axis=0)
            row = loss_rows(logits,truth,cat)
            if method=='risk_mlp':
                row['per_seed'] = {str(seed):loss_rows(logits[i],truth,cat) for i,seed in enumerate(SEEDS)}
            row['by_scenario'] = {s:loss_rows(bykey[f'{method}_{s}'],parts[partition][s]['truth'],phase) for s in SCENARIOS}
            result[partition][method] = row
    return result


def trace_events(truth,eligible,alarm,day_ids,origins,ts):
    columns = ['exposure','events','warned','detected','false','duplicate','late',
               'censored_events','censored_alerts','unknown_alerts']
    daily = np.zeros((int(day_ids.max())+1,len(columns)),np.float64)
    records=[]; leads=[]; delays=[]
    for node in range(truth.shape[1]):
        np.add.at(daily[:,0],day_ids,eligible[:,node]/288.)
        for lo,hi in runs(eligible[:,node]):
            complete=[]; censored=[]
            for a,b in runs(truth[lo:hi,node]):
                s,e=int(lo+a),int(lo+b)
                if s-2<lo or e==hi:
                    censored.append((s,e)); daily[day_ids[s],7]+=1
                else:
                    r={'node':node,'event_start_origin':int(origins[s]),'timestamp':str(ts[origins[s]]),
                       'event_end_origin_exclusive':int(origins[e-1]+1),'proxy_positive_steps':e-s,
                       'status':'missed','alarm_start_origin':None,'lead_minutes':None,
                       'late_minutes':None,'alarm_start_point_label':None,
                       'any_alarm_match_window':bool(alarm[s-2:e,node].any()),
                       'any_alarm_warning_window':bool(alarm[s-2:s+1,node].any()),
                       'active_spanning_warning_window_entry':bool(s-3>=lo and alarm[s-3,node] and alarm[s-2,node])}
                    complete.append((s,e,r)); records.append(r); daily[day_ids[s],1]+=1
            used=set()
            for a,b in runs(alarm[lo:hi,node]):
                a,b=int(a+lo),int(b+lo); d=day_ids[a]
                if a==lo or b==hi:
                    daily[d,8]+=1; continue
                if any(s-2<=a<e for s,e in censored):
                    daily[d,9]+=1; continue
                matches=[j for j,(s,e,_) in enumerate(complete) if s-2<=a<e]
                unmatched=[j for j in matches if j not in used]
                if unmatched:
                    j=unmatched[0]; used.add(j); s,e,r=complete[j]
                    daily[day_ids[s],3]+=1
                    r['alarm_start_origin']=int(origins[a]); r['alarm_start_point_label']=bool(truth[a,node])
                    if a<=s:
                        r['status']='warned'; r['lead_minutes']=5*(s-a+1)
                        daily[day_ids[s],2]+=1; leads.append(r['lead_minutes'])
                    else:
                        r['status']='late'; r['late_minutes']=5*(a-s-1)
                        daily[day_ids[s],6]+=1; delays.append(r['late_minutes'])
                elif matches:
                    daily[d,5]+=1
                else:
                    daily[d,4]+=1
    total=daily.sum(axis=0); count=int(total[1])
    stats={k:float(v) for k,v in zip(columns,total)}
    stats.update({'warning_recall':float(total[2]/count),'detection_recall':float(total[3]/count),
                  'missed_events':int(total[1]-total[3]),'false_per_valid_station_day':float(total[4]/total[0]),
                  'warning_lead_mean_minutes':float(np.mean(leads)) if leads else None,
                  'warning_lead_median_minutes':float(np.median(leads)) if leads else None,
                  'late_detection_mean_minutes':float(np.mean(delays)) if delays else None})
    groups={status:[r for r in records if r['status']==status] for status in ('warned','late','missed')}
    detail={'warning_by_lead':{str(minutes):sum(r['lead_minutes']==minutes for r in groups['warned']) for minutes in (5,10,15)},
            'warning_starts_label_negative':sum(not r['alarm_start_point_label'] for r in groups['warned']),
            'missed_without_any_alarm_in_match_window':sum(not r['any_alarm_match_window'] for r in groups['missed']),
            'missed_with_alarm_but_without_credited_start':sum(r['any_alarm_match_window'] for r in groups['missed']),
            'missed_with_active_spanning_warning_window_entry':sum(r['active_spanning_warning_window_entry'] for r in groups['missed']),
            'duration_by_status':{status:{'count':len(rows),
               'positive_steps_mean':float(np.mean([r['proxy_positive_steps'] for r in rows])) if rows else None,
               'positive_steps_median':float(np.median([r['proxy_positive_steps'] for r in rows])) if rows else None}
               for status,rows in groups.items()}}
    return stats,daily,records,detail


def event_diagnostic(all_logits,parts,data,old_result):
    summaries,daily_all,records_all={},{},[]
    for partition in ('calibration','evaluation'):
        summaries[partition]={}
        for method in METHODS:
            summaries[partition][method]={}
            op=old_result['selection']['methods'][method]['selected']
            for s in SCENARIOS:
                p=parts[partition][s]
                if method=='dense_speed':
                    probability=p['single']*70
                else:
                    values=all_logits[partition][f'{method}_{s}']
                    probability=sigmoid(values) if method=='state_logistic' else sigmoid(values).mean(axis=0)
                alarm=online_alarm(probability,op['threshold'],op['confirmation'])
                stats,daily,records,details=trace_events(p['truth'],p['eligible'],alarm,p['day_ids'],p['origins'],data[3])
                reference,ref_daily,_=event_stats(p['truth'],p['eligible'],alarm,p['day_ids'])
                assert stats==reference and np.array_equal(daily,ref_daily)
                if partition=='evaluation':
                    assert all(v==old_result['evaluation'][method][s][k] for k,v in stats.items())
                summaries[partition][method][s]={'stats':stats,'diagnostic':details}
                daily_all[f'{partition}_{method}_{s}']=daily
                records_all.extend({'partition':partition,'method':method,'scenario':s,**r} for r in records)
    return summaries,daily_all,records_all


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args(); output=args.output; output.mkdir(parents=True,exist_ok=True)
    paths=[Path(__file__),ROOT/'docs/congestion_objective_diagnostic_protocol.zh-CN.md',
           ROOT/'congestion_ablation.py',ROOT/'congestion_risk_pilot.py',ROOT/'congestion_alert_pilot.py',
           ROOT/'state_value_pilot.py',ROOT/'recovery_pilot.py',ROOT/'data/raw/metr-la.h5',
           ROOT/'outputs/recovery-pilot/config.json',ROOT/'outputs/multifault-pilot/config.json']
    paths += [OLD/k for k in ('normalization.npz','features.json','config.json','model_selection.json',
              'results.json','audit.json','logistic.pt','predictions_training.npz','predictions_validation.npz',
              'scores_calibration.npz','scores_evaluation.npz','daily_metrics.npz')]
    paths += [OLD/f'mlp{seed}.pt' for seed in SEEDS]
    paths += [ROOT/'outputs/congestion-alert-pilot-v2'/f'predictions_{part}.npz' for part in ('calibration','evaluation')]
    config={'date':'2026-10-10','exploratory':True,'no_training_or_threshold_change':True,
            'partitions':PARTITIONS,'phases':PHASES,'methods':METHODS,
            'sha256':{str(p.relative_to(ROOT)):sha256(p) for p in paths}}
    config=json.loads(json.dumps(config))
    if (output/'config.json').exists():
        assert config==json.loads((output/'config.json').read_text(encoding='utf-8')),'Use a new output directory'
    else:
        write_json(output/'config.json',config)
    if args.prepare_only:
        print(json.dumps({'config':str(output/'config.json'),'no_training':True})); return
    if (output/'results.json').exists():
        raise ValueError('Existing diagnostic preserved')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); started=time.perf_counter()
    data=load_data(); parts,names=load_parts(data); maps,targets=build_maps(parts,data)
    with np.load(OLD/'normalization.npz') as n:
        mean,std=n['mean'],n['std']
    xt=pool(parts['training'],'state'); checked_std=xt.std(axis=0); checked_std[checked_std<1e-5]=1
    assert np.array_equal(mean,xt.mean(axis=0)) and np.array_equal(std,checked_std)
    all_logits=rebuild_logits(parts,mean,std)
    for partition in ('calibration','evaluation'):
        with np.load(OLD/f'scores_{partition}.npz') as cached:
            for s in SCENARIOS:
                for method in ('state_logistic','risk_mlp'):
                    l=all_logits[partition][f'{method}_{s}']; p=sigmoid(l)
                    if method=='risk_mlp': p=p.mean(axis=0)
                    assert np.array_equal(p,cached[f'{method}_{s}'])
    print('raw phases and frozen probabilities checked; calculating losses and event traces',flush=True)
    losses=loss_diagnostic(all_logits,parts,maps)
    old=json.loads((OLD/'results.json').read_text(encoding='utf-8'))
    event_summary,daily,records=event_diagnostic(all_logits,parts,data,old)
    for partition,values in all_logits.items():
        np.savez_compressed(output/f'logits_{partition}.npz',**values)
    np.savez_compressed(output/'phase_maps.npz',**{f'{name}_{field}':p[field] for name,p in maps.items() for field in ('phase','selected','origins')})
    np.savez_compressed(output/'event_daily_metrics.npz',**daily)
    write_json(output/'event_records.json',records)
    with (output/'event_records.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    result={'config':config,'targets':targets,'loss_diagnostic':losses,'event_diagnostic':event_summary,
            'frozen_operating_points':{m:old['selection']['methods'][m]['selected'] for m in METHODS},
            'elapsed_seconds':time.perf_counter()-started,'new_method_originality_confirmed':False}
    write_json(output/'results.json',result)
    audit={'audit_status':'independent recomputation pending',
           'source_hashes_unchanged':all(sha256(ROOT/k)==v for k,v in config['sha256'].items()),
           'cached_complete_model_probabilities_exactly_equal':True,
           'traced_event_stats_and_daily_arrays_equal_original':True,
           'artifact_sha256':{p.name:sha256(p) for p in output.iterdir() if p.suffix in ('.npz','.csv') or p.name=='event_records.json'}}
    assert audit['source_hashes_unchanged']; write_json(output/'audit.json',audit)
    print(json.dumps({'training_targets':targets['training'],
                     'mlp_training_loss':losses['training']['risk_mlp'],
                     'evaluation_events':event_summary['evaluation'],'seconds':result['elapsed_seconds']},indent=2))


if __name__=='__main__':
    main()
