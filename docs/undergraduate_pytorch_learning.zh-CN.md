# 本课题PyTorch入门与答辩练习

2026-10-13。先掌握“张量如何流动、哪些参数能学习、损失怎样产生梯度”，再讲复杂术语。本材料对应实际代码，配套合成练习不读取真实交通数据、既有权重或缓存，不进行优化器更新。

## 第一步 看懂一批数据

Tensor是带形状的数值数组，形状决定各层怎样计算。B表示一次处理的预测窗口数，不是站点数或整个数据集大小；每个窗口包含24站。

| 位置 | 实际形状 | 每个维度的意思 |
| --- | --- | --- |
| 历史速度和mask | B×12×24 | 窗口、过去12步、24站 |
| 日历 | B×4 | 每个起点的4项已知日历特征 |
| 拼接后 | B×12×24×18 | 6项速度/图/mask、4日历、8嵌入 |
| Conv1d输入 | (B·24)×18×12 | 将每站作为一条序列；18通道，12步 |
| 通道投影 | (B·24)×32×12 | 18输入通道映射为32隐藏通道 |
| 门控卷积输出 | (B·24)×64×12 | 分为32个signal和32个gate |
| 三步预测 | B×3×24 | 每站未来三个位置的速度预测 |
| 风险层输入 | N×51 | N个站时点、51项可见风险特征 |
| 风险MLP内部 | N×51→N×32→N×16→N×1 | 线性层和ReLU |
| 风险logit | N | squeeze(-1)去掉最后的单维 |

`permute`改变维度顺序，`reshape`调整形状；它们在这里把窗口与站点合并给卷积处理，没有增加独立样本。官方[Conv1d说明](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Conv1d.html)解释批次、通道、长度的输入约定。

在合成练习中B=2，因此卷积批次为48，三步输出为2×3×24。风险练习另造48×51的特征；它演示风险层的张量接口，没有实现完整交通特征构造或正式告警评测。

## 第二步 用mask表示能看到什么

原代码先计算 `values = z * mask`。例如标准化速度为0可能表示站点训练均值，也可能只是缺失占位符，必须结合mask解释。这里的合成张量均为有限数，不能把NaN直接乘0当清理；正式读取先根据自然有效性处理非有限值。

图聚合同时计算“邻居可见值的加权和”和“邻居可见权重”，两者相除。没有可见贡献时分子为0，用 `clamp_min(1e-6)`避免除零。固定相关图不等同于地理道路距离，也没有根据当前输入动态学习图边。

合成练习把0号站所有历史mask设为0，再把其隐藏值改为999，输出应逐元素相同。这只能检查该教学调用的隐藏值路径，不代替原正式审核或证明所有信息泄漏都不存在。

## 第三步 分清参数和buffer

`nn.Module`组织网络与子模块；调用 `model(x)`进入forward计算。卷积权重、线性权重及 `nn.Parameter` 的站点嵌入属于学习参数。`register_buffer`保存固定相关图和两跳矩阵，它们随state_dict保存，在本实现中不参与Adam更新。参见[Module官方说明](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Module.html)。

实际图TCN有19523个参数，一份51→32→16→1风险MLP有2209个参数。参数较多不能当效果或原创性证据。24×8的站点嵌入对应当前节点集合，不能仅靠它声称新站点归纳能力。

## 第四步 时间卷积为什么只填左侧

三个时间卷积膨胀率是1、2、4，核大小均为3，分别在左侧补2、4、8个位置。卷积后长度仍为12，避免使用某个历史位置之后的隐藏状态；整个输入窗口又截止在预测起点之前。

每层输出分成signal和gate，然后计算 `tanh(signal) * sigmoid(gate)`，相当于给候选信号设置一个0到1的门，随后加入原隐藏状态。最后取时间轴末位置，经Linear输出三步修正并叠加最近可见速度残差。全部历史不可见时残差为标准化0，最终仍包含网络修正，不应说最终物理速度必为0。

练习hook记录的是卷积实际收到的已填充输入，所以三个时间卷积的长度分别为14、16、20，输出均为12；这和上表的未填充隐藏长度12没有矛盾。

## 第五步 logit 概率 损失和梯度

风险MLP先输出任意实数logit，Sigmoid将其映射到0到1。训练使用 `binary_cross_entropy_with_logits`，直接接收logit与0/1目标；不要先Sigmoid再把概率送给这个函数。可对照[PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.8/generated/torch.nn.BCEWithLogitsLoss.html)。

三种子风险分数是各模型先Sigmoid再取概率均值，不是先平均logit。风险分数经阈值和连续确认才成为告警，不能把分数直接当事件级召回率。

正式训练循环的顺序如下，此处仅讲解原代码，不会执行更新：

```python
optimizer.zero_grad(set_to_none=True)  # 清除上一批梯度
logits = model(x)                     # 正向计算
loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
loss.backward()                      # 计算参数梯度
optimizer.step()                     # 正式训练时才更新参数
```

`backward()`产生梯度，不会自行修改权重；`step()`才按优化器规则更新。PyTorch默认累积梯度，所以每批开始清梯度。合成练习运行一次backward并检查梯度有限、参数未变化，完全不调用step；教学损失值不是论文结果。

预测前端的正式损失是仅在未来自然有效目标上计算标准化L1平均，并在更新前裁剪梯度范数至5。风险层使用BCE，不是同一损失。前端标准化L1不等同于原mph尺度MAE，不能把两个模型的损失单位混写。

## 第六步 区分训练 验证 校准和评测

正式前端三种子42/43/44各训练25轮，按预测验证MAE选权重。正式风险MLP三种子202/203/204各做1350次更新、batch 4096、有放回均匀取样，在225/450/900/1350次检查验证BCE选权重。不要把历史风险代码中的早期30轮fit设置说成最终PEMS设置；最终路径使用 `congestion_density_pilot.py::fit_uniform`。

验证时 `model.eval()`切换如Dropout的行为，`torch.no_grad()`关闭梯度记录；两者不是同一操作。[no_grad官方说明](https://docs.pytorch.org/docs/2.8/generated/torch.no_grad.html)描述梯度记录的关闭机制。先固定预测器，再训练风险层，随后在校准段选阈值，最后回放评测段，各步骤的信息用途不同。

## 运行合成练习

在项目目录执行：

```powershell
& .\.venv\Scripts\python.exe undergraduate_torch_demo.py
```

脚本实例化实际网络类，但使用合成数据、单位矩阵教学图和随机初始化模型，不加载已训练模型。输出写入 `outputs/undergraduate-defense/torch_demo.json`，记录张量形状、隐藏值不变检查、全缺测输出有限检查，以及一次反向传播。已有研究权重与数据不写入、不加载。

已运行检查：输出为2×3×24，风险logit为48，隐藏值更改不影响预测，全缺测输出为有限数，风险参数梯度存在且有限；优化器更新次数为0。此结果只证明这次教学调用能运行，不能证明交通预测精度或新模型优越性。

## 用自己的话回答这六题

1. 为什么2×12×24经过重排会得到48×18×12，18来自哪里？
2. 同样一个标准化数值0，mask=0和mask=1分别是什么意思？
3. 站点embedding和adjacency谁能学习，谁是本实现的固定量？
4. 左侧padding为什么是2、4、8，hook看到的长度为什么不是12？
5. BCEWithLogits为什么不能先Sigmoid；backward为什么没有改变练习权重？
6. 验证损失下降为什么不能直接证明事件告警更好？

答案要点分别见第一至第六步。先做到能解释，再阅读现有[详细训练讲解](pems_bay_pytorch_walkthrough.zh-CN.md)和[答辩讲解稿](undergraduate_defense_notes.zh-CN.md)。

## 实际代码入口

| 讲解内容 | 文件与定义 |
| --- | --- |
| 图TCN、门控时间卷积与前端训练 | `fresh_holdout_pilot.py::MaskedGraphTCN / train_tcn` |
| 最近可见残差和人工遮蔽 | `recovery_pilot.py::latest / artificial_mask` |
| 风险MLP和可见51维特征 | `congestion_risk_pilot.py::RiskModel / features` |
| 最终风险取样与更新预算 | `congestion_density_pilot.py::fit_uniform` |
| PEMS风险准备与概率集成 | `train_pems_bay_risk.py::prepare / scores` |
| 无真实数据的教学调用 | `undergraduate_torch_demo.py::lesson` |

