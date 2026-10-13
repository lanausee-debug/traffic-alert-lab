"""Training-grid density with equal MLP minibatch/update budgets."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
import torch

from recovery_pilot import ROOT,write_json
from multifault_pilot import sha256
from fresh_holdout_pilot import MaskedGraphTCN
from congestion_alert_pilot import SCENARIOS,labels,outage_mask,infer
from congestion_risk_pilot import (RiskModel,prepare_partition,pool,score,sigmoid,choose,
                                  summary_key,hidden_features_check)
from congestion_solver_audit import fit_linear
from congestion_ablation import OLD,SEEDS,load_data,load_parts,load_model
from congestion_onset_pilot import (OUTPUT as ONSET, METHODS as PREVIOUS_METHODS,
    SIMPLE as PREVIOUS_SIMPLE,onset_targets,supervised_data,evaluate)
from congestion_objective_diagnostic import trace_events

OUTPUT=ROOT/'outputs/congestion-density-pilot'
NEW_METHODS=('sparse_update_linear','sparse_update_mlp','dense_update_linear','dense_update_mlp')
METHODS=PREVIOUS_METHODS+NEW_METHODS
SIMPLE=PREVIOUS_SIMPLE+('sparse_update_linear','dense_update_linear')
BATCH=4096
UPDATES=1350
CHECKS=(225,450,900,1350)


def state_signature(state):
    h=hashlib.sha256()
    for name,value in state.items():
        a=value.detach().cpu().numpy()
        h.update(name.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def sampler_signature(rows,seed):
    rng=torch.Generator().manual_seed(seed+17); h=hashlib.sha256()
    for step in range(UPDATES):
        ids=torch.randint(rows,(BATCH,),generator=rng)
        h.update(ids.numpy().astype(np.int32).tobytes())
    return h.hexdigest()


def fit_uniform(x,y,xv,yv,seed,path):
    torch.manual_seed(seed); model=RiskModel(51,True)
    initial=state_signature(model.state_dict())
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.001)
    xt,yt=torch.tensor(x),torch.tensor(y)
    rng=torch.Generator().manual_seed(seed+17); digest=hashlib.sha256()
    logs=[]; best=float('inf'); selected=None; chosen=None
    for step in range(1,UPDATES+1):
        model.train(); ids=torch.randint(len(x),(BATCH,),generator=rng)
        digest.update(ids.numpy().astype(np.int32).tobytes())
        optimizer.zero_grad(set_to_none=True)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(model(xt[ids]),yt[ids])
        loss.backward(); optimizer.step()
        if step in CHECKS:
            logits=score(model,xv)
            value=float(np.mean(np.logaddexp(0,logits)-yv*logits))
            logs.append({'step':step,'validation_bce':value})
            if value<best:
                best=value; chosen=step; selected=copy.deepcopy(model.state_dict())
    model.load_state_dict(selected); model.eval()
    record={'seed':seed,'nonlinear':True,'dimension':51,'training_rows':len(x),'validation_rows':len(xv),
            'training_scheme':'uniform_with_replacement','batch_size':BATCH,'total_updates':UPDATES,
            'training_draws':BATCH*UPDATES,'checkpoint_steps':CHECKS,'selected_step':chosen,
            'validation_bce':best,'logs':logs,'initial_parameters_sha256':initial,
            'minibatch_index_sha256':digest.hexdigest()}
    record=json.loads(json.dumps(record))
    torch.save({'state_dict':selected,**record},path)
    return model,record


def dense_training(data,sparse,output,generate=False):
    raw,z,valid,ts,context,mean,scale,adjacency,day,groups=data
    start,end=int(np.searchsorted(day,42)),int(np.searchsorted(day,70))
    cache=output/'predictions_training_dense.npz'
    generated=False; inference_seconds=0.
    if not cache.exists():
        assert generate,'Missing dense forecast cache'
        begun=time.perf_counter(); origins=np.arange(start+12,end-2)
        shared=np.searchsorted(origins,sparse['natural']['origins'])
        assert np.array_equal(origins[shared],sparse['natural']['origins'])
        missing=np.ones(len(origins),bool); missing[shared]=False
        window=origins[:,None]+np.arange(-12,3)[None]
        regular=(np.diff(ts[window],axis=1)==np.timedelta64(5,'m')).all(axis=1)
        truth,eligible,_=labels(raw,valid,origins,regular)
        dates=np.unique(ts[origins].astype('datetime64[D]'))
        saved={'origins':origins,'truth':truth,'eligible':eligible,
               'day_ids':np.searchsorted(dates,ts[origins].astype('datetime64[D]')),'dates':dates.astype(str)}
        models=[]
        for seed in (42,43,44):
            model=MaskedGraphTCN(12,3,adjacency)
            model.load_state_dict(torch.load(ROOT/'outputs/fresh-holdout-pilot'/f'model_tcn_seed{seed}.pt',weights_only=True)['state_dict'])
            model.eval(); models.append(model)
        with np.load(OLD/'predictions_training.npz') as old:
            for s in SCENARIOS:
                mask,fault=outage_mask(valid,day,s,groups)
                predictions=np.empty((3,len(origins),3,raw.shape[1]),np.float32)
                predictions[:,shared]=old[f'predictions_{s}']
                predictions[:,missing]=infer(models,z,mask,origins[missing],context,mean,scale)
                saved[f'predictions_{s}']=predictions; saved[f'fault_{s}']=fault[origins-1]
                print(f'new dense training forecasts: {s}, elapsed={time.perf_counter()-begun:.1f}s',flush=True)
        np.savez_compressed(cache,**saved); generated=True; inference_seconds=time.perf_counter()-begun
    dense,names=prepare_partition('training_dense',start,end,1,data,[],output)
    shared=np.searchsorted(dense['natural']['origins'],sparse['natural']['origins'])
    maximum=0.
    for s in SCENARIOS:
        for key in ('truth','eligible','fault'):
            assert np.array_equal(dense[s][key][shared],sparse[s][key])
        for key in ('state','single','group'):
            difference=float(np.max(np.abs(dense[s][key][shared]-sparse[s][key])))
            maximum=max(maximum,difference); assert difference<=1e-6
            dense[s][key][shared]=sparse[s][key]
    return dense,names,{'cache_generated':generated,'new_forecast_seconds':inference_seconds,
                       'shared_feature_maximum_difference_before_canonicalization':maximum,
                       'dense_origins':len(dense['natural']['origins']),'shared_sparse_origins':len(shared),
                       'new_origins':len(dense['natural']['origins'])-len(shared)}


def target_for(parts,data):
    day=data[8]; start,end=int(np.searchsorted(day,42)),int(np.searchsorted(day,70))
    t,m=onset_targets(data,parts['natural']['origins'],start,end)
    return {'truth':t,'eligible':m,'origins':parts['natural']['origins']}


def saved_targets(partition):
    with np.load(ONSET/f'targets_{partition}.npz') as f: return {k:f[k] for k in f.files}


def previous_scores(partition):
    with np.load(ONSET/f'scores_{partition}.npz') as f:
        return {m:{s:f[f'{m}_{s}'] for s in SCENARIOS} for m in PREVIOUS_METHODS}


def new_scores(parts,fitted,mean,std):
    result={m:{} for m in fitted}
    for s,p in parts.items():
        x=((p['state'].reshape(-1,51)-mean)/std).astype(np.float32)
        for method,models in fitted.items():
            result[method][s]=np.mean([sigmoid(score(model,x)) for model in models],axis=0).reshape(p['truth'].shape)
    return result


def reference_method(selections):
    return min(SIMPLE,key=lambda m:(*summary_key(list(selections[m]['selected']['by_scenario'].values())),SIMPLE.index(m)))


def evaluate_with_leads(scores,parts,target,selections,ts):
    rows,daily=evaluate(scores,parts,target,selections); records={}
    for method,by_s in scores.items():
        records[method]={}; op=selections[method]['selected']
        for s,probability in by_s.items():
            p=parts[s]
            from congestion_risk_pilot import online_alarm
            alarm=online_alarm(probability,op['threshold'],op['confirmation'])
            stats,core,trace,detail=trace_events(p['truth'],p['eligible'],alarm,p['day_ids'],p['origins'],ts)
            assert all(rows[method][s][k]==v for k,v in stats.items())
            assert np.array_equal(core,daily[f'{method}_{s}'])
            extended=np.pad(core,((0,0),(0,3)))
            for r in trace:
                if r['status']=='warned':
                    index=int(np.searchsorted(p['origins'],r['event_start_origin']))
                    extended[p['day_ids'][index],9+r['lead_minutes']//5]+=1
            assert np.array_equal(extended[:,10:13].sum(axis=1),core[:,2])
            rows[method][s]['warning_lead_counts']=detail['warning_by_lead']
            rows[method][s]['warning_recall_at_least10']=(extended[:,11:13].sum())/stats['events']
            daily[f'{method}_{s}']=extended; records[method][s]=trace
    return rows,daily,records


def paired_success(records,candidate,baseline):
    rows={}; all_deltas=[]
    for s in SCENARIOS:
        a={(r['node'],r['event_start_origin']):r for r in records[candidate][s]}
        b={(r['node'],r['event_start_origin']):r for r in records[baseline][s]}
        assert set(a)==set(b)
        deltas=[a[k]['lead_minutes']-b[k]['lead_minutes'] for k in a if a[k]['status']==b[k]['status']=='warned']
        all_deltas.extend(deltas)
        rows[s]={'shared_warned_events':len(deltas),'mean_paired_lead_difference_minutes':float(np.mean(deltas)) if deltas else None,
                 'difference_counts':{str(d):deltas.count(d) for d in (-10,-5,0,5,10)}}
    return {'candidate':candidate,'baseline':baseline,'by_scenario':rows,'shared_warned_scenario_events':len(all_deltas),
            'mean_paired_lead_difference_minutes':float(np.mean(all_deltas)) if all_deltas else None}


def difference(a,b):
    rng=np.random.default_rng(8120); draws=rng.integers(0,a.shape[1],size=(2000,a.shape[1]))
    ar,br=a[:,draws].sum(axis=2),b[:,draws].sum(axis=2)
    at,bt=a.sum(axis=1),b.sum(axis=1); out={}
    for label,indices,denominator,primary in [('warning10_recall',[11,12],1,True),('warning_recall',[2],1,False),
                                            ('detection_recall',[3],1,False),('false_rate',[4],0,False)]:
        sample=(ar[:,:,indices].sum(axis=-1)/ar[:,:,denominator]-br[:,:,indices].sum(axis=-1)/br[:,:,denominator]).mean(axis=0)
        point=(at[:,indices].sum(axis=-1)/at[:,denominator]-bt[:,indices].sum(axis=-1)/bt[:,denominator]).mean()
        out[label+'_difference']=float(point); out[label+'_ci95']=np.quantile(sample,[.025,.975]).tolist()
        if primary: out[label+'_ci975']=np.quantile(sample,[.0125,.9875]).tolist()
    return out


def comparisons(daily,reference):
    return {key:{'candidate':'dense_update_mlp','baseline':baseline,**difference(
        np.stack([daily[f'dense_update_mlp_{s}'] for s in SCENARIOS]),np.stack([daily[f'{baseline}_{s}'] for s in SCENARIOS]))}
        for key,baseline in [('dense_mlp-sparse_mlp','sparse_update_mlp'),('dense_mlp-calibration_reference',reference)]}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--prepare-only',action='store_true'); args=parser.parse_args(); output=args.output
    output.mkdir(parents=True,exist_ok=True)
    paths=[Path(__file__),ROOT/'docs/congestion_density_protocol.zh-CN.md',ROOT/'congestion_onset_pilot.py',
           ROOT/'congestion_risk_pilot.py',ROOT/'congestion_alert_pilot.py',ROOT/'congestion_ablation.py',
           ROOT/'congestion_solver_audit.py',ROOT/'congestion_objective_diagnostic.py',ROOT/'fresh_holdout_pilot.py',
           ROOT/'state_value_pilot.py',ROOT/'recovery_pilot.py',ROOT/'data/raw/metr-la.h5',
           ROOT/'outputs/recovery-pilot/config.json',ROOT/'outputs/multifault-pilot/config.json']
    paths += [OLD/k for k in ('normalization.npz','features.json','predictions_training.npz','predictions_validation.npz')]
    paths += [ONSET/k for k in ('results.json','audit.json','model_selection.json','onset_linear.pt',
              'scores_calibration.npz','scores_evaluation.npz','daily_metrics.npz')]
    paths += [ONSET/f'targets_{p}.npz' for p in ('training','validation','calibration','evaluation')]
    paths += [ROOT/'outputs/congestion-alert-pilot-v2'/f'predictions_{p}.npz' for p in ('calibration','evaluation')]
    paths += [ROOT/'outputs/fresh-holdout-pilot'/f'model_tcn_seed{s}.pt' for s in (42,43,44)]
    config={'date':'2026-10-10','exploratory':True,'methods':METHODS,'simple_reference_candidates':SIMPLE,
            'seeds':SEEDS,'uniform_batch_size':BATCH,'total_updates':UPDATES,'checkpoint_steps':CHECKS,
            'primary_metric':'warning_recall_at_least10','calibration_selection_unchanged':True,
            'sha256':{str(p.relative_to(ROOT)):sha256(p) for p in paths}}
    config=json.loads(json.dumps(config))
    if (output/'config.json').exists(): assert config==json.loads((output/'config.json').read_text(encoding='utf-8')),'Use a new output directory'
    else: write_json(output/'config.json',config)
    if args.prepare_only:
        print(json.dumps({'config':str(output/'config.json'),'updates':UPDATES,'batch':BATCH})); return
    if (output/'results.json').exists(): raise ValueError('Existing run preserved')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); begun=time.perf_counter()
    data=load_data(); parts,names=load_parts(data)
    dense,names2,forecast_info=dense_training(data,parts['training'],output,generate=True); assert names==names2
    dense_target=target_for(dense,data); sparse_target=saved_targets('training')
    shared=np.searchsorted(dense_target['origins'],sparse_target['origins'])
    assert all(np.array_equal(dense_target[k][shared],sparse_target[k]) for k in ('truth','eligible'))
    np.savez_compressed(output/'targets_training_dense.npz',**dense_target)
    for partition in ('training','validation','calibration','evaluation'): shutil.copy2(ONSET/f'targets_{partition}.npz',output/f'targets_{partition}.npz')
    shutil.copy2(OLD/'normalization.npz',output/'normalization.npz'); write_json(output/'features.json',names)
    with np.load(output/'normalization.npz') as n: mean,std=n['mean'],n['std']
    xs,ys=supervised_data(parts['training'],sparse_target,mean,std)
    xd,yd=supervised_data(dense,dense_target,mean,std)
    xv,yv=supervised_data(parts['validation'],saved_targets('validation'),mean,std)
    old=json.loads((ONSET/'results.json').read_text(encoding='utf-8'))
    old_logs=json.loads((ONSET/'model_selection.json').read_text(encoding='utf-8'))
    assert math.ceil(len(xs)/BATCH)*30==UPDATES
    hidden_features_check(parts['training'],data); hidden_features_check(dense,data)
    fitted={}; logs={}; gradients={}; fit_begin=time.perf_counter()
    for density,x,y in [('sparse',xs,ys['onset']),('dense',xd,yd['onset'])]:
        key=f'{density}_update_linear'
        if density=='sparse':
            shutil.copy2(ONSET/'onset_linear.pt',output/f'{key}.pt'); model,saved=load_model(output/f'{key}.pt'); record=old_logs['onset_linear']
        else: model,record=fit_linear(x,y,xv,yv['onset'],output/f'{key}.pt')
        fitted[key]=[model]; logs[key]=record
        gradients[key]=next(r['max_abs_gradient'] for r in record['logs'] if r['alpha']==record['selected_alpha'])
        key=f'{density}_update_mlp'; fitted[key]=[]; logs[key]=[]
        for seed in SEEDS:
            model,record=fit_uniform(x,y,xv,yv['onset'],seed,output/f'{key}_{seed}.pt')
            fitted[key].append(model); logs[key].append(record)
            print(f'{density} MLP seed={seed}: selected step={record["selected_step"]}; elapsed={time.perf_counter()-fit_begin:.1f}s',flush=True)
    fit_seconds=time.perf_counter()-fit_begin
    for i in range(len(SEEDS)):
        assert logs['sparse_update_mlp'][i]['initial_parameters_sha256']==logs['dense_update_mlp'][i]['initial_parameters_sha256']
    write_json(output/'model_selection.json',logs)
    cal=previous_scores('calibration'); cal.update(new_scores(parts['calibration'],fitted,mean,std))
    selections=copy.deepcopy(old['selection']['methods']); budgets=old['selection']['budgets']
    for method in NEW_METHODS: selections[method]=choose(method,cal[method],parts['calibration'],budgets)
    reference=reference_method(selections)
    selection={'methods':selections,'budgets':budgets,'reference':reference}
    write_json(output/'selection.json',selection)
    np.savez_compressed(output/'scores_calibration.npz',**{f'{m}_{s}':p for m,rows in cal.items() for s,p in rows.items()})
    print(f'calibration saved, reference={reference}; evaluating unchanged event scoring',flush=True)
    test=previous_scores('evaluation'); test.update(new_scores(parts['evaluation'],fitted,mean,std))
    np.savez_compressed(output/'scores_evaluation.npz',**{f'{m}_{s}':p for m,rows in test.items() for s,p in rows.items()})
    rows,daily,records=evaluate_with_leads(test,parts['evaluation'],saved_targets('evaluation'),selections,data[3])
    summary={density:{'training_points':len(x),'positive_points':int(y.sum()),'unique_valid_points':len(x)//3,'unique_positive_points':int(y.sum())//3}
             for density,x,y in [('sparse',xs,ys['onset']),('dense',xd,yd['onset'])]}
    result={'config':config,'training_pools':summary,'forecast_preparation':forecast_info,
            'risk_fitting_seconds':fit_seconds,'reference':reference,'selection':selection,'evaluation':rows,
            'comparisons':comparisons(daily,reference),
            'paired_success':{'dense-sparse':paired_success(records,'dense_update_mlp','sparse_update_mlp'),
                              'dense-reference':paired_success(records,'dense_update_mlp',reference)},
            'elapsed_seconds':time.perf_counter()-begun,'new_method_originality_confirmed':False}
    write_json(output/'results.json',result); np.savez_compressed(output/'daily_metrics.npz',**daily)
    audit={'audit_status':'independent recomputation pending','source_hashes_unchanged':all(sha256(ROOT/k)==v for k,v in config['sha256'].items()),
           'nested_training_labels_and_features_checked':True,'same_mlp_updates_batches_initializations':True,
           'linear_selected_gradients':gradients,'linear_gradient_check_passed':all(v<=1e-4 for v in gradients.values()),
           'artifact_sha256':{p.name:sha256(p) for p in output.iterdir() if p.suffix in ('.npz','.pt') or p.name in ('features.json','selection.json','model_selection.json')}}
    assert audit['source_hashes_unchanged']; write_json(output/'audit.json',audit)
    print(json.dumps({'training_pools':summary,'reference':reference,'comparisons':result['comparisons'],
                      'forecast_seconds':forecast_info['new_forecast_seconds'],'risk_fitting_seconds':fit_seconds},indent=2))


if __name__=='__main__': main()
