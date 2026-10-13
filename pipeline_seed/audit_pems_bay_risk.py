"""Reconstruct PEMS-BAY risk inputs, labels, selection, weights and scores."""
from datetime import datetime, timezone
import numpy as np
import torch
from train_pems_bay_risk import (ROOT, OUTPUT, FORE, METHODS, SEEDS, PARTITIONS,
    verify_forecaster, guard_risk_reads, pooled, load_risk, scores, probability_metrics)
from train_pems_bay_forecaster import read_json, write_json, load_model
from freeze_external_validation import sha256
from pems_bay_reader import SealedPartitionReader
from pems_bay_windows import calendar_features
from congestion_alert_pilot import infer, SCENARIOS, outage_mask
from congestion_risk_pilot import features, score, RiskModel
from congestion_density_pilot import state_signature, sampler_signature


def main():
    verify_forecaster()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    config, result = read_json(OUTPUT/'config.json'), read_json(OUTPUT/'results.json')
    for name, digest in config['source_sha256'].items(): assert sha256(ROOT/name) == digest, name
    for name, digest in result['artifact_sha256'].items(): assert sha256(OUTPUT/name) == digest, name
    design = read_json(ROOT/'configs/pems_bay_validation_v1.json')
    reader = SealedPartitionReader(ROOT/design['dataset']['local_path'], design)
    with np.load(ROOT/'outputs/pems-bay-binding/forecast_training_statistics.npz') as f:
        stats = {k: f[k].copy() for k in ('mean', 'scale', 'adjacency')}
    groups = read_json(ROOT/'outputs/pems-bay-binding/fault_groups.json')
    forecasters = [load_model(seed, stats['adjacency'], FORE/f'model_tcn_seed{seed}.pt') for seed in (42,43,44)]
    caches, interceptions, counts = {}, [], {}
    maximum_prediction_difference = 0.
    with guard_risk_reads(reader, interceptions):
        for forbidden in ('calibration', 'evaluation', 'forecast_train'):
            try: reader.read_partition(forbidden)
            except PermissionError: pass
            else: raise AssertionError(f'Forbidden partition accessible: {forbidden}')
        for partition in PARTITIONS:
            raw64, ts = reader.read_partition(partition)
            raw = raw64.astype(np.float32); valid = (np.isfinite(raw64)&(raw64>0)).astype(np.float32)
            with np.load(OUTPUT/f'{partition}.npz') as f: saved = {k:f[k] for k in f.files}
            caches[partition] = saved
            independent_origins = np.array([t for t in range(12, len(ts)-2)
                if np.all(np.diff(ts[t-12:t+3]) == np.timedelta64(5,'m'))], dtype=np.int64)
            assert np.array_equal(saved['origins'], independent_origins)
            assert np.array_equal(saved['timestamps'], ts)
            origins = saved['origins']
            truth = np.stack([raw64[origins+i] <= 30 for i in range(3)]).all(axis=0)
            eligible = np.stack([valid[origins+i]>0 for i in range(3)]).all(axis=0)
            common = np.zeros_like(eligible); onset = np.zeros_like(truth)
            for j,t in enumerate(origins):
                if t+4 >= len(ts) or np.any(np.diff(ts[t-12:t+5]) != np.timedelta64(5,'m')): continue
                common[j] = (valid[t-1:t+5]>0).all(axis=0)
                yy = np.stack([(raw64[s:s+3] <= 30).all(axis=0) for s in range(t-1,t+3)])
                onset[j] = (yy[1:] & ~yy[:-1]).any(axis=0) & common[j]
            for name,value in (('truth',truth),('eligible',eligible),('common',common),('onset',onset)):
                assert np.array_equal(saved[name],value), (partition,name)
            counts[partition] = {'origins':len(origins), 'common_points':int(common.sum()),
                                 'persistent_positive_points':int((truth&common).sum()),
                                 'onset_positive_points':int(onset.sum())}
            context = calendar_features(ts); days = (ts.astype('datetime64[D]')-reader.d0).astype(int)
            assert np.array_equal(saved['day_indices'], days[origins])
            z = np.where(valid>0,(raw-stats['mean'])/stats['scale'],0).astype(np.float32)
            positions = np.unique(np.linspace(0,len(origins)-1,12,dtype=int))
            subset = origins[positions]
            for scenario in SCENARIOS:
                mask, fault = outage_mask(valid, days, scenario, groups)
                assert np.array_equal(saved[f'{scenario}_fault'],fault[origins-1])
                prediction = infer(forecasters,z,mask,subset,context,stats['mean'],stats['scale'])
                delta = float(np.max(np.abs(prediction-saved[f'{scenario}_predictions'][:,positions])))
                maximum_prediction_difference = max(maximum_prediction_difference,delta)
                assert delta <= 1e-4
                state,single,group,names = features(raw,mask,subset,context,stats['adjacency'],saved[f'{scenario}_predictions'][:,positions])
                for name,value in (('state',state),('single',single),('group',group)):
                    assert np.array_equal(value,saved[f'{scenario}_{name}'][positions]), name
                hidden = raw.copy(); hidden[mask==0] = 10000
                changed = features(hidden,mask,subset,context,stats['adjacency'],saved[f'{scenario}_predictions'][:,positions])
                for a,b in zip((state,single,group),changed[:3]): assert np.array_equal(a,b)
                # One origin at a time: perturb its future without altering any of its history.
                t = int(subset[len(subset)//2]); altered = raw.copy(); altered[t:] = 10000
                prediction1 = infer(forecasters,z,mask,np.array([t]),context,stats['mean'],stats['scale'])
                altered_z = np.where(valid>0,(altered-stats['mean'])/stats['scale'],0).astype(np.float32)
                prediction2 = infer(forecasters,altered_z,mask,np.array([t]),context,stats['mean'],stats['scale'])
                assert np.array_equal(prediction1,prediction2)
                a = features(raw,mask,np.array([t]),context,stats['adjacency'],prediction1)
                b = features(altered,mask,np.array([t]),context,stats['adjacency'],prediction2)
                for first,second in zip(a[:3],b[:3]): assert np.array_equal(first,second)
    assert len(interceptions)==2 and len(reader.access_log)==2
    access = read_json(OUTPUT/'access_log.json')
    assert [r['partition'] for r in access['preparation_reads']] == list(PARTITIONS)
    for p,r in zip(PARTITIONS,access['preparation_reads']):
        assert (r['row_start'],r['row_stop_exclusive'])==reader.bounds[p]
        assert r['columns']==reader.columns.tolist()
    tr,va = [caches[p] for p in PARTITIONS]
    with np.load(OUTPUT/'normalization.npz') as f: norm = {k:f[k] for k in f.files}
    rawx = pooled(tr,'state'); raws = pooled(tr,'single')[:,None]
    expected = {'mean':rawx.mean(axis=0),'std':np.maximum(rawx.std(axis=0),1e-3),
                'single_mean':raws.mean(axis=0),'single_std':np.maximum(raws.std(axis=0),1e-3)}
    for k,v in expected.items(): assert np.array_equal(v,norm[k]), k
    x=((rawx-norm['mean'])/norm['std']).astype(np.float32); del rawx
    xv=((pooled(va,'state')-norm['mean'])/norm['std']).astype(np.float32)
    single=((raws-norm['single_mean'])/norm['single_std']).astype(np.float32)
    sv=((pooled(va,'single')[:,None]-norm['single_mean'])/norm['single_std']).astype(np.float32)
    y={k:np.tile(tr[k][tr['common']],3).astype(np.float32) for k in ('truth','onset')}
    yv={k:np.tile(va[k][va['common']],3).astype(np.float32) for k in ('truth','onset')}
    gt,gv=pooled(tr,'group'),pooled(va,'group')
    logs=read_json(OUTPUT/'model_selection.json')
    fitted={'global_calibration':[], 'group_calibration':{}, 'state_logistic':[],
            'persistent_mlp':[], 'onset_linear':[], 'onset_mlp':[]}
    gradients={}; selected_steps={}
    def audit_model(name,rec,a,b,av,bv,seed=None):
        path=OUTPUT/f'{name}.pt'; model=load_risk(path)
        weight=torch.load(path,weights_only=True)
        assert sha256(path)==rec['weight_sha256'] and sha256(OUTPUT/'config.json')==rec['config_sha256']
        assert rec['training_rows']==len(b) and rec['validation_rows']==len(bv)
        val_logits=score(model,av)
        val_bce=float(np.mean(np.logaddexp(0,val_logits)-bv*val_logits))
        assert abs(val_bce-rec['validation_bce']) <= 2e-6
        chosen=min(rec['logs'],key=lambda r:r['validation_bce'])
        assert chosen['validation_bce']==rec['validation_bce']
        if seed is None:
            assert chosen['alpha']==rec['selected_alpha']==weight['selected_alpha']
            assert chosen['max_abs_gradient']<=1e-4
            model.zero_grad(set_to_none=True)
            objective=torch.nn.functional.binary_cross_entropy_with_logits(model(torch.from_numpy(a)),torch.from_numpy(b))
            objective=objective+rec['selected_alpha']/2*model.layers.weight.square().sum()
            objective.backward()
            grad=max(float(p.grad.abs().max()) for p in model.parameters())
            assert grad<=1e-4 and abs(grad-chosen['max_abs_gradient'])<=1e-6
            gradients[name]=grad
        else:
            assert chosen['step']==rec['selected_step']==weight['selected_step']
            assert [r['step'] for r in rec['logs']]==[225,450,900,1350]
            assert rec['actual_optimizer_updates']==rec['total_updates']==1350
            assert rec['batch_size']==4096 and rec['training_draws']==5529600
            torch.manual_seed(seed); initial=RiskModel(51,True)
            assert state_signature(initial.state_dict())==rec['initial_parameters_sha256']
            assert sampler_signature(len(b),seed)==rec['minibatch_index_sha256']
            selected_steps[name]=rec['selected_step']
        return model
    fitted['global_calibration']=[audit_model('global_calibration',logs['global_calibration'],single,y['truth'],sv,yv['truth'])]
    for g in (0,1,2):
        a,b=gt==g,gv==g; rec=logs['group_calibration'][str(g)]
        reasons=[]
        if a.sum()<100: reasons.append('training_rows_below_100')
        if len(np.unique(y['truth'][a]))<2: reasons.append('one_training_class')
        if not b.any(): reasons.append('empty_validation')
        if reasons:
            assert rec['reasons']==reasons and rec['fallback']=='global_calibration'
        else:
            assert 'fallback' not in rec
            fitted['group_calibration'][str(g)]=[audit_model(f'group{g}',rec,single[a],y['truth'][a],sv[b],yv['truth'][b])]
    for method,obj in (('state_logistic','truth'),('onset_linear','onset')):
        fitted[method]=[audit_model(method,logs[method],x,y[obj],xv,yv[obj])]
    for method,obj in (('persistent_mlp','truth'),('onset_mlp','onset')):
        for seed,rec in zip(SEEDS,logs[method]):
            fitted[method].append(audit_model(f'{method}_seed{seed}',rec,x,y[obj],xv,yv[obj],seed))
    for a,b in zip(logs['persistent_mlp'],logs['onset_mlp']):
        for k in ('seed','initial_parameters_sha256','minibatch_index_sha256'): assert a[k]==b[k]
    for partition,part in caches.items():
        rebuilt=scores(part,fitted,norm)
        with np.load(OUTPUT/f'scores_{partition}.npz') as f:
            assert sorted(f.files)==sorted(rebuilt)
            for k,v in rebuilt.items(): assert np.array_equal(v,f[k]),k
        if partition=='risk_validation':
            for method in METHODS[1:]:
                obj='onset' if method.startswith('onset_') else 'truth'
                panel='onset_target_Z' if obj=='onset' else 'persistent_target_Y'
                p=np.concatenate([rebuilt[f'{method}__{s}'][part['common']] for s in SCENARIOS])
                assert probability_metrics(p,yv[obj])==result['validation'][panel][method]
    audit={'passed':True,'audited_at_utc':datetime.now(timezone.utc).isoformat(),
           'risk_training_executed':True,'calibration_executed':False,'evaluation_values_read':False,
           'checks':{'source_and_artifact_sha256':True,'independent_Y_Z_and_common_pool':True,
                     'independent_time_gap_windows':True,'training_only_normalization':True,
                     'actual_optimizer_budget_and_matching_initialization_sampler':True,
                     'selected_BCE_and_linear_convergence':True,'all_seven_method_scores_exact':True,
                     'hidden_and_future_speed_perturbation_invariance':True,'forbidden_partition_denial':True},
           'partition_counts':counts,'selected_linear_max_abs_gradient':gradients,
           'selected_mlp_updates':selected_steps,'forecast_sample_maximum_difference_mph':maximum_prediction_difference,
           'reader':reader.access_log,'runtime_interceptions':interceptions,
           'results_sha256':sha256(OUTPUT/'results.json')}
    if (OUTPUT/'audit.json').exists():
        old=read_json(OUTPUT/'audit.json'); audit['audited_at_utc']=old['audited_at_utc']
        assert audit==old
    else: write_json(OUTPUT/'audit.json',audit)
    print({'passed':True,'counts':counts,'gradients':gradients,'evaluation_values_read':False})


if __name__=='__main__': main()
