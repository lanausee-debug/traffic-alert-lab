"""One fixed-setting PEMS-BAY evaluation; no fitting, selection or grid scan."""
from datetime import datetime,timezone
import time
import numpy as np
import torch
from freeze_pems_bay_evaluation import ROOT,OUTPUT,SEAL,verify_execution
from pems_bay_evaluation_gate import EvaluationReader,guard_evaluation_reads
from calibrate_pems_bay_alerts import frozen_scores,METHODS,GRIDS
from freeze_pems_bay_evaluation import CAL
from train_pems_bay_forecaster import read_json,write_json,load_model
from freeze_external_validation import sha256
from congestion_alert_pilot import SCENARIOS
from pems_bay_alert_replay import alarm,replay,layout
import train_pems_bay_risk as risk


METRICS=(('warning10_recall',(11,12),1),('warning_recall',(2,),1),
         ('detection_recall',(3,),1),('false_rate',(4,),0))


def paired_difference(candidate,baseline,draws,primary=False):
    if candidate.shape!=baseline.shape or candidate.shape[0]!=3 or candidate.shape[-1]!=13:
        raise ValueError('Paired date-count array mismatch')
    if draws.shape!=(2000,candidate.shape[1]): raise ValueError('Bootstrap dates must match the frozen calendar')
    ct,bt=candidate.sum(axis=1),baseline.sum(axis=1)
    cs,bs=candidate[:,draws,:].sum(axis=2),baseline[:,draws,:].sum(axis=2)
    result={}
    for name,columns,denominator in METRICS:
        point_valid=bool(np.all(ct[:,denominator]>0)&np.all(bt[:,denominator]>0))
        point=float(np.mean(ct[:,columns].sum(axis=-1)/ct[:,denominator]-bt[:,columns].sum(axis=-1)/bt[:,denominator])) if point_valid else None
        invalid=np.any((cs[:,:,denominator]<=0)|(bs[:,:,denominator]<=0),axis=0)
        if point_valid and not invalid.any():
            samples=(cs[:,:,columns].sum(axis=-1)/cs[:,:,denominator]-bs[:,:,columns].sum(axis=-1)/bs[:,:,denominator]).mean(axis=0)
            ci=np.quantile(samples,[.025,.975]).tolist()
            ci975=np.quantile(samples,[.0125,.9875]).tolist() if primary and name=='warning10_recall' else None
        else: ci=None; ci975=None
        result[name]={'difference':point,'ci95':ci,'undefined_bootstrap_replicates':int(invalid.sum()),
                      'interval_status':'computed' if ci is not None else 'undefined_denominator'}
        if primary and name=='warning10_recall': result[name]['ci975']=ci975
    return result


def comparison_specs():
    specs=[('onset_mlp_fine_minus_global_fine','onset_mlp','fine','global_calibration','fine',True),
           ('onset_mlp_fine_minus_group_fine','onset_mlp','fine','group_calibration','fine',True),
           ('onset_mlp_fine_minus_persistent_fine','onset_mlp','fine','persistent_mlp','fine',False),
           ('onset_mlp_fine_minus_onset_linear_fine','onset_mlp','fine','onset_linear','fine',False)]
    return specs+[(f'{m}_fine_minus_coarse',m,'fine',m,'coarse',False) for m in METHODS]


def compare_all(daily,draws):
    result={}
    for key,cm,cg,bm,bg,primary in comparison_specs():
        a=np.stack([daily[f'{cm}__{cg}__{s}'] for s in SCENARIOS])
        b=np.stack([daily[f'{bm}__{bg}__{s}'] for s in SCENARIOS])
        result[key]={'candidate':f'{cm}_{cg}','baseline':f'{bm}_{bg}','primary':primary,
                     'metrics':paired_difference(a,b,draws,primary)}
    return result


def decision(comparisons,metrics):
    primary=[row['metrics']['warning10_recall'].get('ci975') for row in comparisons.values() if row['primary']]
    executable=all(row['events']>0 and row['exposure']>0 for bygrid in metrics.values() for bys in bygrid.values() for row in bys.values())
    supported=bool(executable and len(primary)==2 and all(ci is not None and ci[0]>0 for ci in primary))
    return {'executable':executable,'both_primary_ci975_lower_bounds_positive':supported,
            'primary_metric_supported':supported,'overall_service_improvement_established':False,
            'algorithm_originality_established':False,
            'evaluation_data_analyzed_after_this_run':True}


def main():
    seal=verify_execution(); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    proof=read_json(OUTPUT/'preflight.json')
    assert proof['passed'] and proof['execution_freeze_sha256']==sha256(SEAL)
    if (OUTPUT/'results.json').exists():
        prior=read_json(OUTPUT/'results.json')
        assert prior['execution_freeze_sha256']==sha256(SEAL)
        for name,digest in prior['artifact_sha256'].items(): assert sha256(OUTPUT/name)==digest,name
        print('Completed fixed-setting evaluation verified; no re-evaluation or selection'); return
    begun=time.perf_counter(); design=read_json(ROOT/'configs/pems_bay_validation_v1.json')
    reader=EvaluationReader(ROOT/design['dataset']['local_path'],design,permit=True)
    interceptions=[]
    with guard_evaluation_reads(reader,interceptions):
        cache=OUTPUT/'evaluation.npz'; prep=OUTPUT/'preparation.json'; started=OUTPUT/'unseal_started.json'
        if cache.exists() and prep.exists():
            info=read_json(prep)
            assert info['execution_freeze_sha256']==sha256(SEAL) and info['cache_sha256']==sha256(cache)
            with np.load(cache) as f: part={k:f[k] for k in f.files}
        elif cache.exists() or prep.exists() or started.exists():
            raise ValueError('Unseal already started without a complete verified cache; preserve partial artifacts')
        else:
            write_json(started,{'started_at_utc':datetime.now(timezone.utc).isoformat(),'stage':'first_evaluation_speed_access_authorized',
                'execution_freeze_sha256':sha256(SEAL),'preflight_sha256':sha256(OUTPUT/'preflight.json'),
                'selection_sha256':sha256(CAL/'selection.json'),'evaluation_bounds':seal['evaluation_bounds']})
            with np.load(ROOT/'outputs/pems-bay-binding/forecast_training_statistics.npz') as f:
                stats={k:f[k].copy() for k in ('mean','scale','adjacency')}
            models=[load_model(seed,stats['adjacency'],risk.FORE/f'model_tcn_seed{seed}.pt') for seed in (42,43,44)]
            groups=read_json(ROOT/'outputs/pems-bay-binding/fault_groups.json')
            part,names,summary=risk.prepare(reader,'evaluation',models,stats,groups)
            np.savez_compressed(cache,**part)
            info={'summary':summary,'feature_names':names,'execution_freeze_sha256':sha256(SEAL),
                  'cache_sha256':sha256(cache),'reader':reader.access_log[-1],'runtime_interception':interceptions[-1]}
            write_json(prep,info)
        write_json(OUTPUT/'access_log.json',{'reader_this_invocation':reader.access_log,'runtime_this_invocation':interceptions,
                  'preparation_read':info['reader'],'evaluation_values_read':True,
                  'execution_freeze_sha256':sha256(SEAL)})
        score_started=time.perf_counter(); scores=frozen_scores(part)
        np.savez_compressed(OUTPUT/'scores_evaluation.npz',**scores)
        scoring_seconds=time.perf_counter()-score_started
        selection=read_json(CAL/'selection.json')
        selected={m:{g:selection['methods'][m][g]['selected'] for g in GRIDS} for m in METHODS}
        assert selected==seal['selected_settings']
        write_json(OUTPUT/'selected_settings.json',{'settings':selected,'selection_source_sha256':sha256(CAL/'selection.json')})
        metrics={}; daily={}; records={}; exceeded={}; replay_started=time.perf_counter()
        for method in METHODS:
            metrics[method]={}; exceeded[method]={}
            for grid in GRIDS:
                metrics[method][grid]={}; exceeded[method][grid]={}
                setting=selected[method][grid]
                for s in SCENARIOS:
                    bits=alarm(scores[f'{method}__{s}'],setting['threshold'],setting['confirmation'],part)
                    row,days,events=replay(part,bits,True,s)
                    metrics[method][grid][s]=row
                    key=f'{method}__{grid}__{s}'; daily[key]=days; records[key]=events
                    rate=row['false_per_valid_station_day']
                    exceeded[method][grid][s]={'calibration_budget':selection['budgets'][s],'evaluation_rate':rate,
                        'exceeded':bool(rate is not None and rate>selection['budgets'][s]+1e-12)}
            print(f'Fixed evaluation replay completed: {method}',flush=True)
        replay_seconds=time.perf_counter()-replay_started
        np.savez_compressed(OUTPUT/'daily_metrics.npz',**daily); write_json(OUTPUT/'event_records.json',records)
        with np.load(OUTPUT/'bootstrap_draws.npz') as f:
            draws=f['draws']; assert np.array_equal(f['dates'],layout(part)[2])
        comparisons=compare_all(daily,draws); verdict=decision(comparisons,metrics)
    verify_execution()
    artifacts=('execution_freeze.json','bootstrap_draws.npz','preflight.json','unseal_started.json','evaluation.npz',
               'preparation.json','access_log.json','scores_evaluation.npz','selected_settings.json','daily_metrics.npz','event_records.json')
    result={'completed_at_utc':datetime.now(timezone.utc).isoformat(),'stage':'one_fixed_setting_held_out_evaluation',
            'execution_freeze_sha256':sha256(SEAL),'evaluation_values_read':True,'held_out_evaluation_executed':True,
            'models_retrained':False,'evaluation_threshold_grid_scanned':False,'replayed_scenario_operating_points':42,
            'evaluation_dates':layout(part)[2].astype(str).tolist(),'evaluation_segments':len(layout(part)[0]),
            'evaluation':metrics,'comparisons':comparisons,'verdict':verdict,'calibration_budget_exceedance':exceeded,
            'costs_seconds':{'preparation':info['summary']['preparation_seconds'],'risk_scoring_and_save':scoring_seconds,
                              'fixed_replay':replay_seconds,'this_completion_invocation':time.perf_counter()-begun},
            'artifact_sha256':{name:sha256(OUTPUT/name) for name in artifacts}}
    write_json(OUTPUT/'results.json',result)
    print({'evaluation':'completed','verdict':verdict,'primary_comparisons':{k:v for k,v in comparisons.items() if v['primary']}},flush=True)


if __name__=='__main__': main()
