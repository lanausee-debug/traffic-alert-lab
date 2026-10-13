"""Controlled persistent-low-speed versus impending-onset supervision."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch

from recovery_pilot import ROOT,write_json
from multifault_pilot import sha256
from congestion_alert_pilot import SCENARIOS,point_stats
from congestion_risk_pilot import (METHODS as OLD_METHODS,BASELINES as OLD_BASELINES,
    fit,score,sigmoid,choose,summary_key,operating,online_alarm,pool,hidden_features_check)
from congestion_solver_audit import fit_linear
from congestion_ablation import OLD,SEEDS,load_data,load_parts
from congestion_ablation import contrast

OUTPUT=ROOT/'outputs/congestion-onset-pilot'
PARTITIONS=[('training',42,70,3),('validation',70,77,1),('calibration',77,91,1),('evaluation',91,119,1)]
OBJECTIVES=('common_persistent','onset')
NEW_METHODS=tuple(f'{objective}_{family}' for objective in OBJECTIVES for family in ('linear','mlp'))
METHODS=OLD_METHODS+NEW_METHODS
SIMPLE=OLD_BASELINES+('common_persistent_linear','onset_linear')


def onset_targets(data,origins,start,end):
    raw,_,valid,ts,*_=data
    positions=np.minimum(origins[:,None]+np.arange(-1,5)[None],end-1)
    values=raw[positions]
    known=(valid[positions]>0).all(axis=1)
    history=np.minimum(origins[:,None]+np.arange(-12,5)[None],end-1)
    regular=(np.diff(ts[history],axis=1)==np.timedelta64(5,'m')).all(axis=1)
    inside=(origins-12>=start)&(origins+4<end)
    known &= (regular & inside)[:,None]
    new_starts=np.stack([(values[:,i]>30)&(values[:,i+1:i+4]<=30).all(axis=1) for i in range(3)],axis=1)
    target=new_starts.any(axis=1)&known
    return target,known


def make_targets(parts,data):
    targets,summaries={},{}
    day=data[8]
    for partition,a,b,stride in PARTITIONS:
        p=parts[partition]['natural']; start,end=int(np.searchsorted(day,a)),int(np.searchsorted(day,b))
        target,known=onset_targets(data,p['origins'],start,end)
        assert not (known & ~p['eligible']).any()
        old=p['truth']; valid_count=int(known.sum())
        targets[partition]={'truth':target,'eligible':known,'origins':p['origins']}
        summaries[partition]={'unique_original_valid_points':int(p['eligible'].sum()),
            'unique_common_valid_points':valid_count,'pooled_common_points':valid_count*len(SCENARIOS),
            'common_fraction_of_original_valid':valid_count/int(p['eligible'].sum()),
            'unique_persistent_positive_points':int((old & known).sum()),
            'unique_onset_positive_points':int(target.sum()),
            'both_positive':int((old & target & known).sum()),
            'persistent_only_positive':int((old & ~target & known).sum()),
            'onset_only_positive':int((~old & target & known).sum()),
            'both_negative':int((~old & ~target & known).sum()),
            'boundary_excluded_origins':int((p['origins']+4>=end).sum()),
            'unknown_points_excluded':int((~known).sum())}
    return targets,summaries


def supervised_data(parts,target,mean,std):
    mask=target['eligible']
    x=np.concatenate([parts[s]['state'][mask] for s in SCENARIOS])
    x=((x-mean)/std).astype(np.float32)
    labels={'common_persistent':np.concatenate([parts[s]['truth'][mask] for s in SCENARIOS]).astype(np.float32),
            'onset':np.tile(target['truth'][mask],len(SCENARIOS)).astype(np.float32)}
    return x,labels


def new_scores(parts,fitted,mean,std):
    result={m:{} for m in NEW_METHODS}
    for s,p in parts.items():
        x=((p['state'].reshape(-1,51)-mean)/std).astype(np.float32)
        for method,models in fitted.items():
            result[method][s]=np.mean([sigmoid(score(m,x)) for m in models],axis=0).reshape(p['truth'].shape)
    return result


def old_scores(partition):
    with np.load(OLD/f'scores_{partition}.npz') as f:
        return {m:{s:f[f'{m}_{s}'] for s in SCENARIOS} for m in OLD_METHODS}


def reference_method(selections):
    def rank(method):
        return (*summary_key(list(selections[method]['selected']['by_scenario'].values())),SIMPLE.index(method))
    return min(SIMPLE,key=rank)


def probability_metrics(probability,truth,eligible):
    y=truth[eligible].astype(np.float64); p=probability[eligible]
    return {'brier':float(np.mean((p-y)**2)),
            'bce':float(np.mean(-y*np.log(np.clip(p,1e-7,1-1e-7))-(1-y)*np.log(np.clip(1-p,1e-7,1)))),
            'valid_points':int(eligible.sum()),'positive_points':int((truth & eligible).sum())}


def evaluate(scores,parts,target,selections):
    summaries,daily_all={},{}
    for method,by_s in scores.items():
        summaries[method]={}; op=selections[method]['selected']
        for s,probability in by_s.items():
            p=parts[s]; row,daily=operating(probability,p,op['threshold'],op['confirmation'])
            alarm=online_alarm(probability,op['threshold'],op['confirmation'])
            row['onset_auxiliary']={'point':point_stats(target['truth'],target['eligible'],alarm)}
            if method!='dense_speed':
                row['persistent_probability_auxiliary']=probability_metrics(probability,p['truth'],p['eligible'])
                row['onset_auxiliary'].update(probability_metrics(probability,target['truth'],target['eligible']))
            summaries[method][s]=row; daily_all[f'{method}_{s}']=daily
    return summaries,daily_all


def comparisons(daily,reference):
    result={}
    for candidate,baseline,key,primary in [
        ('onset_mlp','common_persistent_mlp','onset_mlp-common_persistent_mlp',True),
        ('onset_mlp',reference,'onset_mlp-calibration_reference',True),
        ('onset_linear','common_persistent_linear','onset_linear-common_persistent_linear',False),
        ('onset_mlp','onset_linear','onset_mlp-onset_linear',False)]:
        a=np.stack([daily[f'{candidate}_{s}'] for s in SCENARIOS])
        b=np.stack([daily[f'{baseline}_{s}'] for s in SCENARIOS])
        result[key]={'candidate':candidate,'baseline':baseline,**contrast(a,b,primary)}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT); parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args(); output=args.output; output.mkdir(parents=True,exist_ok=True)
    paths=[Path(__file__),ROOT/'docs/congestion_onset_protocol.zh-CN.md',ROOT/'congestion_ablation.py',
           ROOT/'congestion_risk_pilot.py',ROOT/'congestion_solver_audit.py',ROOT/'congestion_alert_pilot.py',
           ROOT/'state_value_pilot.py',ROOT/'recovery_pilot.py',ROOT/'data/raw/metr-la.h5',
           ROOT/'outputs/recovery-pilot/config.json',ROOT/'outputs/multifault-pilot/config.json']
    paths += [OLD/k for k in ('normalization.npz','features.json','config.json','model_selection.json','results.json',
              'audit.json','predictions_training.npz','predictions_validation.npz','scores_calibration.npz',
              'scores_evaluation.npz','daily_metrics.npz')]
    paths += list(OLD.glob('*.pt'))
    paths += [ROOT/'outputs/congestion-alert-pilot-v2'/f'predictions_{p}.npz' for p in ('calibration','evaluation')]
    config={'date':'2026-10-10','exploratory':True,'all_dates_previously_used':True,
            'partitions':PARTITIONS,'methods':METHODS,'simple_reference_candidates':SIMPLE,
            'seeds':SEEDS,'input_dimension':51,'training_stride_minutes':15,
            'common_training_mask':True,'event_evaluation_unchanged':True,
            'sha256':{str(p.relative_to(ROOT)):sha256(p) for p in paths}}
    config=json.loads(json.dumps(config))
    if (output/'config.json').exists():
        assert config==json.loads((output/'config.json').read_text(encoding='utf-8')),'Use a new output directory'
    else: write_json(output/'config.json',config)
    if args.prepare_only:
        print(json.dumps({'config':str(output/'config.json'),'methods':METHODS,'new_checkpoints':8})); return
    if (output/'results.json').exists(): raise ValueError('Existing experiment preserved')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); started=time.perf_counter()
    data=load_data(); parts,names=load_parts(data); targets,target_summary=make_targets(parts,data)
    for partition in ('training','validation'): hidden_features_check(parts[partition],data)
    for partition,t in targets.items(): np.savez_compressed(output/f'targets_{partition}.npz',**t)
    shutil.copy2(OLD/'normalization.npz',output/'normalization.npz')
    write_json(output/'features.json',names)
    with np.load(output/'normalization.npz') as n: mean,std=n['mean'],n['std']
    x,ys=supervised_data(parts['training'],targets['training'],mean,std)
    xv,yvs=supervised_data(parts['validation'],targets['validation'],mean,std)
    fitted,logs,gradients={},{},{}
    for objective in OBJECTIVES:
        key=f'{objective}_linear'
        model,record=fit_linear(x,ys[objective],xv,yvs[objective],output/f'{key}.pt')
        fitted[key]=[model]; logs[key]=record
        gradients[key]=next(r['max_abs_gradient'] for r in record['logs'] if r['alpha']==record['selected_alpha'])
        key=f'{objective}_mlp'; fitted[key]=[]; logs[key]=[]
        for seed in SEEDS:
            model,record=fit(x,ys[objective],xv,yvs[objective],True,seed,output/f'{key}_{seed}.pt')
            fitted[key].append(model); logs[key].append(record)
        print(f'{objective}: 1 linear + 3 MLP checkpoints selected; elapsed={time.perf_counter()-started:.1f}s',flush=True)
    write_json(output/'model_selection.json',logs)
    cal=old_scores('calibration'); cal.update(new_scores(parts['calibration'],fitted,mean,std))
    old=json.loads((OLD/'results.json').read_text(encoding='utf-8'))
    selections=copy.deepcopy(old['selection']['methods']); budgets=old['selection']['budgets']
    for method in NEW_METHODS: selections[method]=choose(method,cal[method],parts['calibration'],budgets)
    reference=reference_method(selections)
    write_json(output/'selection.json',{'methods':selections,'budgets':budgets,'reference':reference})
    np.savez_compressed(output/'scores_calibration.npz',**{f'{m}_{s}':p for m,rows in cal.items() for s,p in rows.items()})
    print(f'calibration choices saved; simple reference={reference}; evaluating frozen settings',flush=True)
    eval_scores=old_scores('evaluation'); eval_scores.update(new_scores(parts['evaluation'],fitted,mean,std))
    np.savez_compressed(output/'scores_evaluation.npz',**{f'{m}_{s}':p for m,rows in eval_scores.items() for s,p in rows.items()})
    rows,daily=evaluate(eval_scores,parts['evaluation'],targets['evaluation'],selections)
    differences=comparisons(daily,reference)
    result={'config':config,'target_summary':target_summary,'reference':reference,
            'selection':{'methods':selections,'budgets':budgets,'reference':reference},
            'evaluation':rows,'comparisons':differences,'elapsed_seconds':time.perf_counter()-started,
            'new_method_originality_confirmed':False}
    write_json(output/'results.json',result); np.savez_compressed(output/'daily_metrics.npz',**daily)
    audit={'audit_status':'independent recomputation pending',
           'source_hashes_unchanged':all(sha256(ROOT/k)==v for k,v in config['sha256'].items()),
           'hidden_and_future_features_invariant':True,'same_common_training_samples':True,
           'linear_selected_gradients':gradients,'linear_gradient_check_passed':all(v<=1e-4 for v in gradients.values()),
           'artifact_sha256':{p.name:sha256(p) for p in output.iterdir() if p.suffix in ('.npz','.pt') or p.name in ('features.json','selection.json','model_selection.json')}}
    assert audit['source_hashes_unchanged']; write_json(output/'audit.json',audit)
    print(json.dumps({'target_summary':target_summary,'reference':reference,
                     'primary_comparisons':{k:v for k,v in differences.items() if v['primary']},
                     'gradients_ok':audit['linear_gradient_check_passed'],'seconds':result['elapsed_seconds']},indent=2))


if __name__=='__main__': main()
