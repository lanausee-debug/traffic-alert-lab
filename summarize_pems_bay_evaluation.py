"""Report every fixed held-out result and its predeclared service limits."""
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from freeze_pems_bay_evaluation import ROOT,OUTPUT,SEAL,verify_execution,CAL
from train_pems_bay_forecaster import read_json,write_json
from freeze_external_validation import sha256
from calibrate_pems_bay_alerts import METHODS,GRIDS
from congestion_alert_pilot import SCENARIOS


def means(rows):
    keys=('warning_recall_at_least10','warning_recall','detection_recall','false_per_valid_station_day')
    return {k:float(np.mean([rows[s][k] for s in SCENARIOS])) if all(rows[s][k] is not None for s in SCENARIOS) else None for k in keys}


def fmt(value,scale=1,digits=2):
    return '不可计算' if value is None else f'{value*scale:.{digits}f}'


def ci_text(values,scale=100):
    return '不可计算' if values is None else f'[{values[0]*scale:.2f}, {values[1]*scale:.2f}]'


def main():
    seal=verify_execution(); result=read_json(OUTPUT/'results.json'); audit=read_json(OUTPUT/'audit.json')
    assert audit['passed'] and audit['results_sha256']==sha256(OUTPUT/'results.json')
    for name,digest in result['artifact_sha256'].items(): assert sha256(OUTPUT/name)==digest,name
    freeze_path=OUTPUT/'evaluation_results_freeze.json'
    if freeze_path.exists():
        prior=read_json(freeze_path)
        for name,digest in prior['source_sha256'].items(): assert sha256(ROOT/name)==digest,name
        print({'held_out_results_freeze':'verified_no_rewrite','verdict':result['verdict']}); return
    prep=read_json(OUTPUT/'preparation.json')['summary']; selected=read_json(OUTPUT/'selected_settings.json')['settings']
    verdict=result['verdict']; sample=result['evaluation'][METHODS[0]]['fine'][SCENARIOS[0]]
    lines=['# PEMS-BAY一次独立评测与论文结论边界', '',
           '日期：2026-10-11。2017-04-09–06-30的83日评测已首次解封，七方法各粗/细已选设置一次回放。所有模型、阈值、预算、计分代码及主比较在读值前冻结；没有在评测段扫描阈值、重训或挑站。评测从此已分析，后续修改不得继续称这段为未见确认数据。', '',
           f'固定24站，{prep["origins"]}个连续在线起点，{int(sample["events"])}个完整原始低速代理事件、{sample["exposure"]:.6f}有效站日，'+str(result['evaluation_segments'])+'个连续时间段。三情景共享自然事件真值与曝光，不是三倍独立样本。事件三读数有效性与风险监督六读数mask保持分开。', '',
           '## 预定主比较', '',
           '主指标为原代理计分的≥10分钟事件预警召回；候选固定起点MLP细网格，分别减全局/分组细网格，两项97.5%区间下界都>0才支持主指标。下表单位为百分点。95%为描述性区间，不能替代预定97.5%判据。', '',
           '| 主比较 | ≥10分钟召回差 | 97.5%区间 | 描述性95%区间 | 下界>0 |', '| --- | ---: | --- | --- | --- |']
    for name,row in result['comparisons'].items():
        if not row['primary']: continue
        metric=row['metrics']['warning10_recall']; ci=metric['ci975']
        lines.append(f'| {row["candidate"]} − {row["baseline"]} | {fmt(metric["difference"],100)} | {ci_text(ci)} | {ci_text(metric["ci95"])} | {"是" if ci is not None and ci[0]>0 else "否"} |')
    lines += ['', '预定主指标判据：**'+('通过' if verdict['primary_metric_supported'] else '未通过')+'**。这不等于整体服务改善、现场适用或算法原创。', '',
              '## 全部方法的独立评测', '', '三情景率等权平均；前3个指标为百分数，误报为告警段数/有效站日。每方法粗细设置均已在校准段选择；评测不再优化。', '',
              '| 方法 | 网格 | 冻结阈值/确认 | ≥10分钟召回 | 全部预警 | 含迟报检测 | 误报/站日 | 超预算情景数 |',
              '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for method in METHODS:
        for grid in GRIDS:
            row=means(result['evaluation'][method][grid]); setting=selected[method][grid]
            t=setting['threshold']; label=f'{-t:g}mph' if method=='dense_speed' else f'{t:g}'
            exceed=sum(r['exceeded'] for r in result['calibration_budget_exceedance'][method][grid].values())
            lines.append(f'| {method} | {grid} | {label}/c{setting["confirmation"]} | {fmt(row["warning_recall_at_least10"],100)} | {fmt(row["warning_recall"],100)} | {fmt(row["detection_recall"],100)} | {fmt(row["false_per_valid_station_day"],digits=6)} | {exceed}/3 |')
    lines += ['', '预算为校准经验约束，评测不保证满足；即使超预算也保持已选设置并照实报告。主指标成功不能掩盖检测下降或误报上升/超预算。', '',
              '## 同一配对日期下的服务差异', '', '| 比较 | 检测差（pp）及95%区间 | 误报差/站日及95%区间 |', '| --- | --- | --- |']
    for row in result['comparisons'].values():
        if not row['primary']: continue
        detect=row['metrics']['detection_recall']; false=row['metrics']['false_rate']
        lines.append(f'| {row["candidate"]} − {row["baseline"]} | {fmt(detect["difference"],100)} {ci_text(detect["ci95"])} | {fmt(false["difference"],digits=6)} {ci_text(false["ci95"],1)} |')
    lines += ['', '误报差为负表示候选误报较少；检测差为负表示更多事件漏检。区间包含零不证明等价或非劣。本方案未规定现场容忍界值，不能把任一单项显著性转换成“整体更好”。', '',
              '## 次要比较', '', '| 次要比较 | ≥10分钟差（pp） | 描述性95%区间 |', '| --- | ---: | --- |']
    for row in result['comparisons'].values():
        if row['primary']: continue
        metric=row['metrics']['warning10_recall']
        lines.append(f'| {row["candidate"]} − {row["baseline"]} | {fmt(metric["difference"],100)} | {ci_text(metric["ci95"])} |')
    lines += ['', '次要比较不替代主比较，不按结果将辅助发现升级成确认性贡献。起点线性是充分收敛的强简单对照，即使起点MLP优于全局/分组，也必须说明是否优于起点线性。', '',
              '## 细网格各情景完整服务记录', '',
              '| 方法 | 情景 | ≥10分钟% | 全部预警% | 检测% | 误报/站日 | 预算 | 超出 | 5/10/15成功 | 漏检 | 重复 | 删失事件/告警 | 未知 |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | --- | ---: |']
    for method in METHODS:
        for s,row in result['evaluation'][method]['fine'].items():
            budget=result['calibration_budget_exceedance'][method]['fine'][s]
            counts='/'.join(str(row['warning_lead_counts'][str(i)]) for i in (5,10,15))
            lines.append(f'| {method} | {s} | {fmt(row["warning_recall_at_least10"],100)} | {fmt(row["warning_recall"],100)} | {fmt(row["detection_recall"],100)} | {fmt(row["false_per_valid_station_day"],digits=6)} | {budget["calibration_budget"]:.6f} | {"是" if budget["exceeded"] else "否"} | {counts} | {row["missed_events"]} | {int(row["duplicate"])} | {int(row["censored_events"])}/{int(row["censored_alerts"])} | {int(row["unknown_alerts"])} |')
    lines += ['', '全部粗细点级覆盖、故障站/其他站结果、平均/中位提前量及迟报、日级计数和一对一事件记录在results.json/daily_metrics.npz/event_records.json。提前5/10/15来自5×(s−a+1)原代理计分，不是经独立事故记录验证的物理起点。', '',
              '## 配对区间、执行与审核', '',
              '83个相同日历日期跨情景/方法联合重采样2000次，seed8120；先合计事件/成功/曝光/误报，再逐情景计算率并等权平均。索引在解封前仅凭时间戳生成并冻结。区间条件于固定模型，不包含重训不确定性；日间相关、少事件和一个网络24站子集限制外推。', '',
              '任何复制出现零分母均不平滑或重抽，对应区间标不可计算；undefined_bootstrap_replicates逐指标保存。原始事件/曝光为0时可执行判据失败，并保持记录，不改方案。', '',
              f'解封前完整依赖SHA冻结{len(seal["source_sha256"])}文件，preflight零评测速度读取通过；生产一次evaluation切片，独立审核再读取同段复核，均至多每实例一次。解封门是协作逻辑约束，不是OS/加密保护。', '',
              f'解封后独立审核通过：原始Y/三读数有效性/连续窗口，冻结预测和特征抽样（最大预测差{audit["forecast_sample_max_difference_mph"]:.3g}mph），全部风险分数、隐藏/未来值/监督mask不变性，全部42固定回放的计数/日级数组/事件记录、11比较×2000复制与判据、超预算标记及全部解封前依赖不变。没有调用拟合/阈值选择器或评测网格扫描。', '',
              '耗时：'+', '.join(f'{k}={v:.1f}秒' for k,v in result['costs_seconds'].items())+'。前阶段训练及扫描成本见已冻结预测/风险/校准日志，不把评测推理耗时当训练耗时。', '',
              '## 论文可写与下一步', '',
              '该结果可以用于已固定流程在另一未分析网络的独立验证：公开数据、信息边界、起点监督、强简单对照、预算/阈值与事件级服务取舍。不能把既有TCN/MLP/逻辑回归或网格扫描写成首次新算法，不宣称真实事件地面真值、现场收益、全325站或零样本跨网络迁移。', '',
              '下一步整理论文实验章节、方法/强对照/负结果与局限，以及可复现GitHub仓库说明和PyTorch教程。评测已完成，不在这83日继续调参后冒充同一次确认。原始交通数据默认不打包公开；星数及毕业创新要求需分别评估，不能由本次统计判据自动保证。']
    (OUTPUT/'report.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Speed','Global','Grouped','State LR','Persistent MLP','Onset LR','Onset MLP']
    fig,axes=plt.subplots(2,1,figsize=(11.5,8),constrained_layout=True); x=np.arange(7)
    if verdict['executable']:
        for i,(key,label,color) in enumerate([('warning_recall_at_least10','Warning >=10 min','#12697c'),
                 ('warning_recall','All warnings','#4799ac'),('detection_recall','Including late detection','#a5c7d2')]):
            axes[0].bar(x+(i-1)*.24,[means(result['evaluation'][m]['fine'])[key]*100 for m in METHODS],.23,label=label,color=color)
        axes[0].set_ylim(0,100); axes[0].legend(ncol=3,loc='lower left',bbox_to_anchor=(0,1.01),fontsize=8)
        axes[0].set_ylabel('Recall (%)')
        for i,(s,color) in enumerate(zip(SCENARIOS,('#3a7d44','#ce9443','#824e8d'))):
            axes[1].bar(x+(i-1)*.24,[result['evaluation'][m]['fine'][s]['false_per_valid_station_day'] for m in METHODS],.23,label=s,color=color)
            axes[1].axhline(result['calibration_budget_exceedance'][METHODS[0]]['fine'][s]['calibration_budget'],color=color,linestyle='--',linewidth=1)
        axes[1].set_ylabel('False alerts / valid station-day'); axes[1].legend(ncol=3,loc='lower left',bbox_to_anchor=(0,1.01),fontsize=8)
    else: axes[0].text(.1,.5,'Frozen evaluation infeasible: zero events or exposure',transform=axes[0].transAxes)
    for ax in axes:
        ax.set_xticks(x,labels); ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
    fig.suptitle('PEMS-BAY: 83-day held-out evaluation | fixed fine-grid settings\nCalibration false-alert budgets shown as dashed lines',fontsize=12)
    for ext in ('png','svg'): fig.savefig(OUTPUT/f'summary.{ext}',dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.2),constrained_layout=True)
    mainrows=[v for v in result['comparisons'].values() if v['primary']]
    for i,row in enumerate(mainrows):
        point=row['metrics']['warning10_recall']['difference']; ci=row['metrics']['warning10_recall']['ci975']
        if point is not None and ci is not None:
            ax.plot(np.array(ci)*100,[i,i],color='#12697c',linewidth=2); ax.plot(point*100,i,'o',color='#12697c')
    ax.set_yticks([0,1],['Onset MLP - global calibration','Onset MLP - grouped calibration'])
    ax.axvline(0,color='gray',linestyle='--'); ax.set_ylim(-.7,1.7); ax.invert_yaxis()
    ax.set_xlabel('Difference in warning >=10 min recall (percentage points)')
    ax.set_title('Predeclared primary comparisons | 97.5% paired-date bootstrap intervals',fontsize=10)
    ax.spines[['top','right','left']].set_visible(False); ax.grid(axis='x',alpha=.15)
    for ext in ('png','svg'): fig.savefig(OUTPUT/f'primary_comparisons.{ext}',dpi=160)
    plt.close(fig)
    paths=[OUTPUT/'results.json',OUTPUT/'audit.json',OUTPUT/'report.zh-CN.md',OUTPUT/'summary.png',OUTPUT/'summary.svg',
           OUTPUT/'primary_comparisons.png',OUTPUT/'primary_comparisons.svg',Path(__file__).resolve()]
    write_json(freeze_path,{'stage':'held_out_evaluation_completed_and_independently_audited',
        'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'execution_freeze_sha256':sha256(SEAL),
        'evaluation_values_read':True,'evaluation_data_now_analyzed':True,'verdict':verdict,
        'source_sha256':{p.relative_to(ROOT).as_posix():sha256(p) for p in paths}})
    print({'held_out_results_freeze':'created','verdict':verdict,
           'fine_results':{m:means(result['evaluation'][m]['fine']) for m in METHODS}},flush=True)


if __name__=='__main__': main()
