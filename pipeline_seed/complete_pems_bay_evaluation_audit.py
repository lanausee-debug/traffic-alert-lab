"""Recorded post-run float-list comparison repair; no evaluation result changes."""
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
import numpy as np
import audit_pems_bay_evaluation as audit
from freeze_pems_bay_evaluation import ROOT,OUTPUT,SEAL,verify_execution,CAL
from train_pems_bay_forecaster import read_json,write_json
from freeze_external_validation import sha256


def recursive_numeric_compare(a,b):
    if isinstance(a,dict):
        assert isinstance(b,dict) and set(a)==set(b)
        for k in a: recursive_numeric_compare(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert isinstance(b,(list,tuple)) and len(a)==len(b)
        for first,second in zip(a,b): recursive_numeric_compare(first,second)
    elif isinstance(a,float):
        assert b is not None and np.isclose(a,b,rtol=1e-12,atol=1e-10),(a,b)
    else: assert a==b,(a,b)


def main():
    verify_execution()
    path=OUTPUT/'numeric_audit_comparator_repair.json'; audit_path=OUTPUT/'audit.json'
    sources={'adapter':sha256(Path(__file__)),'frozen_audit':sha256(ROOT/'audit_pems_bay_evaluation.py'),
             'execution_freeze':sha256(SEAL),'evaluation_results':sha256(OUTPUT/'results.json'),
             'calibration_selection':sha256(CAL/'selection.json')}
    if path.exists():
        record=read_json(path); assert record['preserved_source_sha256']==sources
    else:
        assert not audit_path.exists()
        write_json(path,{'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
            'stage':'post_unseal_audit_implementation_repair_only',
            'reason':'Original assert_same tolerates scalar float rounding but compares CI lists with exact Python equality. Independent exposure accumulation differs at about 1e-14 in a false-rate CI.',
            'example_reference_ci':[-0.1300684751541042,-0.07352022431224627],
            'example_independent_ci':[-0.1300684751540859,-0.0735202243122359],
            'change':'Apply the already-used scalar float tolerance recursively to list elements. Integer counts, shapes, keys, labels, settings and records retain exact comparisons.',
            'float_absolute_tolerance':1e-10,'float_relative_tolerance':1e-12,
            'production_results_or_models_or_settings_or_verdict_changed':False,'evaluation_data_already_analyzed':True,
            'preserved_source_sha256':sources})
    record=read_json(path)
    if audit_path.exists():
        proof=read_json(audit_path)
        assert proof['passed'] and proof['numeric_comparator_repair_sha256']==sha256(path)
        assert proof['audit_adapter_sha256']==sources['adapter'] and proof['results_sha256']==sources['evaluation_results']
        print('Existing independent evaluation audit and recorded comparator repair verified'); return
    with patch.object(audit,'assert_same',recursive_numeric_compare): audit.post_audit()
    assert sha256(OUTPUT/'results.json')==sources['evaluation_results']
    assert sha256(CAL/'selection.json')==sources['calibration_selection']
    verify_execution()
    proof=read_json(audit_path)
    proof['numeric_comparator_repair_sha256']=sha256(path)
    proof['audit_adapter_sha256']=sources['adapter']
    proof['frozen_audit_source_sha256']=sources['frozen_audit']
    proof['production_results_changed_by_audit_repair']=False
    write_json(audit_path,proof)
    (OUTPUT/'numeric_audit_repair.zh-CN.md').write_text(
        '# 评测审核的浮点列表比较修复\n\n'
        '独立审核完成全部42回放的事件计数/记录及风险分数后，置信区间列表严格相等检查失败：误报区间[-0.1300684751541042, -0.07352022431224627]与独立累计曝光所得[-0.1300684751540859, -0.0735202243122359]相差约1e−14。\n\n'
        '保留解封前冻结的审核源码，新增完成脚本；将审核原来已对标量使用的abs=1e−10/rel=1e−12容差递归用于列表浮点元素。计数、结构、标签、已选设置及事件记录仍严格核对。修复后重新独立审核全部42回放及11比较×2000次抽样并通过。\n\n'
        '此为解封后的审核实现修复，已经明确记录，不能伪称该适配在看结果前冻结。原始结果/区间/主判据、模型、校准选择、执行冻结及129依赖SHA均未改变；主验收仍未通过，没有把95%替代97.5%或改变显著性门槛。详见numeric_audit_comparator_repair.json及audit.json。\n',encoding='utf-8')


if __name__=='__main__': main()
