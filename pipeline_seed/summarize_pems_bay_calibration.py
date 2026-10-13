"""Calibrated settings and service trade-offs; no held-out results or claims."""
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from calibrate_pems_bay_alerts import ROOT,OUTPUT,METHODS,GRIDS,verify_risk
from train_pems_bay_forecaster import read_json,write_json
from freeze_external_validation import sha256
from congestion_alert_pilot import SCENARIOS

LABELS=['Speed','Global','Grouped','State LR','Persistent MLP','Onset LR','Onset MLP']


def mean_metrics(rows):
    names=('warning_recall_at_least10','warning_recall','detection_recall','false_per_valid_station_day')
    return {k:float(np.mean([rows[s][k] for s in SCENARIOS])) for k in names}


def setting(method,record):
    t=record['threshold']; c=record['confirmation']
    if t>1 and method!='dense_speed' or method=='dense_speed' and t==1000: return '关闭'
    return (f'{-t:g} mph' if method=='dense_speed' else f'{t:g}')+f' / c{c}'


def main():
    verify_risk()
    result=read_json(OUTPUT/'results.json'); selection=read_json(OUTPUT/'selection.json'); audit=read_json(OUTPUT/'audit.json')
    assert audit['passed'] and audit['results_sha256']==sha256(OUTPUT/'results.json')
    for name,digest in read_json(OUTPUT/'config.json')['source_sha256'].items(): assert sha256(ROOT/name)==digest,name
    for name,digest in result['artifact_sha256'].items(): assert sha256(OUTPUT/name)==digest,name
    paths=[OUTPUT/'config.json',OUTPUT/'results.json',OUTPUT/'audit.json',Path(__file__).resolve()]
    freeze_path=OUTPUT/'calibration_implementation_freeze.json'
    if freeze_path.exists():
        prior=read_json(freeze_path)
        for name,digest in prior['source_sha256'].items(): assert sha256(ROOT/name)==digest,name
        print({'calibration_stage_freeze':'verified_no_rewrite','evaluation_values_read':False}); return
    lines=['# PEMS-BAY独立分区告警校准', '', '日期：2026-10-11。本步骤已完成七方法共同预算、粗细网格和确认次数选择并审核。以下全为校准描述，尚未读取或评测2017-04-09至06-30的83日数据。', '',
           f'校准日期2017-03-26–04-08共14天，{int(result["complete_calibration_events"])}个完整自然低速代理事件、{result["valid_station_days"]:.6f}有效站日。人工故障不改事件真值及曝光；三情景不是三批独立事件。实际校准时间连续，共'+str(result['calibration_segments'])+'段。', '',
           '## 固定预算', '', '| 输入情景 | 速度30mph/c1误报段 | 误报预算/有效站日 |', '| --- | ---: | ---: |']
    budgets=read_json(OUTPUT/'budgets.json')
    for s in SCENARIOS:
        lines.append(f'| {s} | {int(budgets["rows"][s]["false"])} | {result["budgets"][s]:.6f} |')
    lines += ['', '预算仅由校准段固定速度规则产生，对七方法共同约束；没有放宽，即使预算为0也不变。阈值/确认在三情景共用。预算不是未来误报保证，测试时可能超出。', '',
              '## 已选择的运行点和服务指标', '',
              '选择目标依次为最大平均全部预警召回、最大最差情景全部预警召回、最小平均误报，再确认次数和声明网格索引。没有按≥10分钟优化。下表三情景率等权平均，百分数列为召回；“c2”需连续两次激活，缺口重置，午夜不重置。', '',
              '| 方法 | 网格 | 阈值/确认 | ≥10分钟召回 | 全部预警召回 | 含迟报检测 | 误报/站日 |',
              '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for method in METHODS:
        for grid in GRIDS:
            row=mean_metrics(result['selected_service_metrics'][method][grid])
            choice=selection['methods'][method][grid]['selected']
            lines.append(f'| {method} | {grid} | {setting(method,choice)} | {100*row["warning_recall_at_least10"]:.2f}% | {100*row["warning_recall"]:.2f}% | {100*row["detection_recall"]:.2f}% | {row["false_per_valid_station_day"]:.6f} |')
    lines += ['', '细网格包含粗网格，并按同一选择目标择优；这不保证≥10分钟召回、检测或误报各自都改善。不能把本表校准优势当作83日独立测试结论，固定主比较仍是起点MLP细网格分别减全局和分组细网格。', '',
              '## 细网格各情景明细', '',
              '| 方法 | 情景 | ≥10分钟 | 全部预警 | 检测 | 误报/站日 | 5/10/15分钟成功 | 漏检 | 重复 | 删失事件/告警 | 未知告警 |',
              '| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: |']
    for method in METHODS:
        for s,row in result['selected_service_metrics'][method]['fine'].items():
            counts='/'.join(str(row['warning_lead_counts'][str(i)]) for i in (5,10,15))
            lines.append(f'| {method} | {s} | {100*row["warning_recall_at_least10"]:.2f}% | {100*row["warning_recall"]:.2f}% | {100*row["detection_recall"]:.2f}% | {row["false_per_valid_station_day"]:.6f} | {counts} | {row["missed_events"]} | {int(row["duplicate"])} | {int(row["censored_events"])}/{int(row["censored_alerts"])} | {int(row["unknown_alerts"])} |')
    lines += ['', '提前档位采用原5×(s−a+1)计分公式，不是独立现场事故起点。完整全部覆盖、点级故障站/其他站表现、提前/迟报统计、删失、日级数组和事件记录见results.json与event_records.json。未把六读数监督mask替代三读数事件分母。', '',
              '## 执行与审核', '',
              f'2588个方法×网格×阈值×确认组合、三情景共7764记录均完整枚举；概率粗17/细180、速度粗20/细92。阈值扫描及输出保存{result["threshold_scan_seconds"]:.1f}秒，完整准备/评分/扫描{result["elapsed_seconds"]:.1f}秒；没有重训。', '',
              '审核独立逐站重建完整/删失事件和新告警段，一对一匹配全部7764运行点的13列日级计数、预算可行性、选点排序、粗细共享点、42个已选情景运行点及完整事件记录。只实际读取一次校准24站切片；其他分区及完整速度矩阵读取拒绝。重推预测抽样最大差'+f'{audit["forecast_sample_max_difference_mph"]:.3g} mph。', '',
              '合成边界验证覆盖时间缺口中断确认/事件、跨午夜保持、自然未知和删失、无完整事件的段、重复/迟报、一段误报随更高阈值裂成两段、负mph分数方向。实际校准无缺口，合成验证不是新交通实证。所有模型、风险标准化及原阶段SHA核对通过。', '',
              '## 下一步', '', '新增一次评测回放及解封门，冻结完整实现/选择/模型/数据绑定依赖并核对后，首次读取83日评测速度；每方法只回放粗细网格已选点，不扫描评测阈值。按冻结两主比较和同日历配对bootstrap输出97.5%主区间、95%描述区间及完整服务取舍。若主结论失败，保存负结果，不继续用该尾段调整后宣称同一次验证成功。']
    (OUTPUT/'report.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=np.arange(len(METHODS)); fig,axes=plt.subplots(2,1,figsize=(11.5,7.4),constrained_layout=True)
    for i,(key,label,color) in enumerate([('warning_recall_at_least10','Warning >=10 min','#12697c'),
        ('warning_recall','All warnings','#4799ac'),('detection_recall','Including late detection','#a5c7d2')]):
        values=[mean_metrics(result['selected_service_metrics'][m]['fine'])[key]*100 for m in METHODS]
        axes[0].bar(x+(i-1)*.24,values,width=.23,label=label,color=color)
    axes[0].set_ylabel('Recall (%)'); axes[0].legend(ncol=3,loc='upper left',fontsize=9)
    axes[0].set_title('Selected fine-grid alerts | calibration data only',loc='left',weight='bold')
    for i,(s,color) in enumerate(zip(SCENARIOS,('#3a7d44','#ce9443','#824e8d'))):
        values=[result['selected_service_metrics'][m]['fine'][s]['false_per_valid_station_day'] for m in METHODS]
        axes[1].bar(x+(i-1)*.24,values,width=.23,label=s,color=color)
        axes[1].axhline(result['budgets'][s],color=color,linestyle='--',linewidth=1,alpha=.7)
    axes[1].set_ylabel('False alerts / valid station-day'); axes[1].legend(ncol=3,fontsize=8)
    axes[1].set_title('Shared per-scenario empirical budgets (dashed lines)',loc='left')
    for ax in axes:
        ax.set_xticks(x,LABELS); ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
    fig.suptitle('PEMS-BAY: 24 stations, 14-day calibration | 83-day evaluation remains sealed',fontsize=12)
    for ext in ('png','svg'): fig.savefig(OUTPUT/f'summary.{ext}',dpi=160)
    plt.close(fig)
    paths += [OUTPUT/'report.zh-CN.md',OUTPUT/'summary.png',OUTPUT/'summary.svg']
    freeze={'stage':'calibration_complete_and_audited_not_evaluation_unlock',
            'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'calibration_executed':True,
            'evaluation_values_read':False,'evaluation_unlock_implemented':False,
            'source_sha256':{p.relative_to(ROOT).as_posix():sha256(p) for p in paths}}
    write_json(freeze_path,freeze)
    print({'calibration_stage_freeze':'passed','events':result['complete_calibration_events'],
           'selected_fine':{m:mean_metrics(result['selected_service_metrics'][m]['fine']) for m in METHODS},
           'evaluation_values_read':False})


if __name__=='__main__': main()
