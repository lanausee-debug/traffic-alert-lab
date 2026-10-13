# GitHub复现说明与发布准备

日期：2026-10-11。当前材料在本地，尚未创建GitHub仓库或上传。项目题目仍为《面向观测缺失的交通拥堵告警风险校准与事件级决策方法研究》。本页区分结果核对、原完整检查点复核和新训练，三者所需材料不同。

更新：统一从零训练入口已在独立目录完成18阶段本机实跑，预测/风险验证、全部校准设置、评测与区间同原结果相符，差异0项；各阶段执行合计约22分钟。新入口、恢复与验收见[从零运行说明](pems_bay_pipeline_reproduction.zh-CN.md)。以下保留旧轻量v1包的范围与限制；新版`traffic-alert-lab-pipeline-v1`另附源码种子和运行证据。原轻量包本身仍不包含完整训练入口所需种子，不能仅给原包添加原始数据就当完整训练包。

## 1. 轻量结果包：无需交通数据或PyTorch

本地打包脚本`prepare_github_materials.py`生成`publication/traffic-alert-lab-v1`。包内包含冻结结果JSON、选择与SHA记录、论文草稿及完整附表、PEMS-BAY管线相关源码、现有PyTorch教程和已生成图片。它不包含原始HDF5、模型权重、预测/特征缓存或全部历史实验。

从该目录运行：

```powershell
python build_thesis_experiment_draft.py --check
```

该命令仅使用Python标准库，按包内冻结结果重新计算14个运行点、21个细网格情景、两主比较与九辅助比较的表格文字，核对生成清单和文件一致性。它不解码交通速度、不调用模型、不重新抽样或调阈值。通过意味着本包内结果与表格相符，不意味着独立重新验证原始交通事件，也不证明来源SHA记录的真实性。预期输出包含`documentation: verified`及`primary_metric_supported: false`。

修改草稿时，先修改`docs/thesis_experiment_discussion.template.zh-CN.md`，再运行无参数生成命令。数字仍取自原JSON；新文档清单属于写作产物记录，不是新的独立实验。检查命令使用`--check`，只读已有文档。

## 2. 已有完整本地检查点：核对全部冻结依赖

完整检查点保留原数据、权重、缓存与各阶段记录，环境不在ZIP中。进入原项目目录，在与记录一致的环境中运行：

```powershell
& .\.venv\Scripts\python.exe freeze_pems_bay_evaluation.py --verify
& .\.venv\Scripts\python.exe complete_pems_bay_evaluation_audit.py
& .\.venv\Scripts\python.exe build_thesis_experiment_draft.py --check --full-integrity
```

第一条核对解封前129项依赖，可能以字节方式读取HDF5计算SHA，但不解码评测速度。第二条在完整审核已存在时核对审核及数值修复记录，不重新执行全部独立事件审核。第三条核对执行、结果及事后修复/接续冻结，并核对论文表格。轻量包缺少数据与权重，不能执行完整依赖核对；不得为了让检查通过而删除旧冻结里的依赖项。

原生产评测完整完成后的`evaluate_pems_bay_alerts.py`与汇总入口也有已有产物保护。本次文档工作不删除输出、重跑评测或改动原始冻结。

## 3. 旧轻量包的从头训练限制（历史说明；新入口已单独完成）

当前源码展示实际训练过程，但还不是已验证的全平台一键训练产品。部分阶段使用固定输出路径，冻结文件绑定本次输入、源码和产物SHA。轻量包包含相关原源码供阅读，缺少若干完整阶段文件、模型和缓存，不能声称下载一个HDF5后所有原冻结检查就自动通过。

原运行环境为Windows、Python3.12、CPU PyTorch2.8.0，依赖见`requirements-pilot.txt`。若建立独立重训工作目录，可以安装对应环境：

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-pilot.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

安装会访问包源；本轮没有重新安装。环境固定减少差异，但不同硬件和库版本不保证逐位一致。新的完整重训入口需先把所有阶段的数据/输出路径参数化，补全下载、阶段产物和运行状态管理，在独立目录验证一条从零运行链；保留当前冻结目录及负结果。此项尚未完成，不在README中写“完整训练已一键复现”。

既有训练顺序为来源绑定与读取审核→三种子图TCN训练/验证→Y/Z风险模型训练/验证→告警校准全网格审核→执行冻结与预审→一次固定设置评测→独立审核与汇总。沿用当前已分析的83日尾段训练或挑阈值属于后续探索，不能称新确认。

## 4. 源码和PyTorch的阅读路线

| 任务 | 源码或文档 | 重点 |
| --- | --- | --- |
| 数据分区与窗口 | `pems_bay_reader.py`、`pems_bay_windows.py` | 命名分区、连续历史、未来输入边界 |
| 预测网络 | `fresh_holdout_pilot.py`中的`MaskedGraphTCN` | 可见值/比例、固定图buffer、嵌入、因果Conv1d |
| 预测训练 | `train_pems_bay_forecaster.py` | TensorDataset、DataLoader、有效目标L1、Adam、选权重 |
| 风险训练 | `train_pems_bay_risk.py`及`congestion_density_pilot.py` | 51维特征、BCE、相同训练预算、三种子概率平均 |
| 事件服务 | `pems_bay_alert_replay.py`、`calibrate_pems_bay_alerts.py` | 连续确认、事件一对一匹配、曝光、预算约束 |
| 独立验证 | `freeze_pems_bay_evaluation.py`、`audit_pems_bay_evaluation.py` | 事前冻结、日期配对区间、完整负结果 |
| 教程 | `docs/pems_bay_pytorch_walkthrough.zh-CN.md` | 张量形状、backward、train/eval、no_grad、state_dict |

原预测教程保存于预测阶段，末尾“后续训练风险与评测”为当时计划；最新状态以本页和独立评测报告为准。保留原文件以维持冻结SHA。

## 5. 已实现的结果与发布边界

主候选起点MLP细网格长提前量召回11.89%，检测28.18%，误报0.086581次/有效站日。相对全局主差+2.97pp、97.5%区间[−0.20,5.89]；相对分组+3.42pp、[0.36,6.29]。双下界>0的预定主验收未通过，且对起点逻辑回归的辅助长提前量差异未建立。README、图和论文都必须保留这些结论。

打包与上传是不同步骤。当前包用于本地审阅；公开发布前仍需核对保留源码、数据说明和引用材料的归属与再分发条件，并决定自有代码许可。当前没有LICENSE授权声明、公开数据/权重下载页或GitHub地址，不制造这些链接。轻量包不含原始数据，原完整检查点ZIP不应直接作为公开仓库附件。

后续最能增加项目使用价值的工作，是提供可靠的从零运行入口、小型可解释演示和清楚的指标说明。源码可运行与用户能理解结果是可检查目标；GitHub星数受传播与需求影响，不能预先保证，也不能用星数证明论文创新。
