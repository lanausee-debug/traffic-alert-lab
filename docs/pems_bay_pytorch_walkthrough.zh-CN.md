# 这次图TCN训练实际用了哪些PyTorch内容

代码入口：`train_pems_bay_forecaster.py`管理来源、分区和训练记录；`fresh_holdout_pilot.py`中的`MaskedGraphTCN`定义网络、`train_tcn`执行优化；`pems_bay_windows.py`把单分区数据变成训练窗口。

## 一批数据在网络中怎样流动

24站、过去12步、未来3步。一个完整batch的历史速度`x`为`[128,12,24]`，自然有效mask形状相同；未来目标`y`及其有效mask为`[128,3,24]`，日历为`[128,4]`。最后一个batch可以不足128，不能把batch数说成总样本数。

历史索引为`t−12`到`t−1`，未来监督为`t`到`t+2`。当前时刻日历是已知输入，未来真实速度只是训练答案。TensorDataset保存五种张量，DataLoader同步取同一组索引，再按固定种子打乱。

原始速度先减该站训练均值、除训练标准差。无效位置的归一化值填0，另保留mask；因此数值0既可能是真实的训练均值，也可能是占位符，网络要依赖mask区别。

## nn.Module、图聚合和时间卷积

`nn.Module`管理可训练参数与固定buffer。节点嵌入`nn.Parameter`是24×8的可学习站点表示；`register_buffer`保存相关性图及其平方，随state_dict保存，但不由Adam更新。本站ID嵌入依赖这24站，因此换网络要重新训练。

`torch.einsum`用固定图聚合一跳/两跳邻居的可见速度，并除以可用邻居权重。输入缺失后，邻居速度和邻居覆盖都能改变，隐藏真实值不能参与聚合。这是固定相关图聚合，并非图注意力网络。

每站每步有6个速度/mask/图特征，加4个日历和8个嵌入，共18通道。网络将`[B,12,24,18]`重排为`[B×24,18,12]`，供Conv1d处理。`Conv1d(18,32,1)`映射通道，后三个卷积使用膨胀率1/2/4，输出分成signal与gate，组成`tanh(signal)×sigmoid(gate)`。

卷积只在左侧填充，因此不使用序列未来；这里整个输入序列本来就只含真实时刻t之前的12步。时间卷积最后位置映射成3步速度修正，叠加“最近可见速度”的持久性残差。模型最终输出`[B,3,24]`，不是0/1拥堵概率；概率要由后续风险层学习。

## 训练循环中的五个动作

```python
optimizer.zero_grad(set_to_none=True)
prediction = model(x, artificial_mask, calendar)
loss = ((prediction - y).abs() * target_valid).sum() / target_valid.sum().clamp_min(1)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
optimizer.step()
```

清梯度是因为PyTorch默认累积梯度；`backward()`沿计算图求损失对参数的导数；梯度截断限制这次更新前的梯度范数；Adam利用梯度和内部累积状态调整参数。固定图buffer仍不训练。

训练人工mask从自然有效mask出发，再隐藏部分站点历史与零散读数。它只改变模型能看到什么，不把被隐藏位置的未来答案也删掉。目标自然无效才不计损失。归一化L1的单位不是mph；验证时要乘回各站标准差才能报告mph MAE。

## train、eval、no_grad和选权重

训练时`model.train()`开启Dropout；验证时`model.eval()`关闭Dropout，`torch.no_grad()`不建立反向传播图。两者作用不同：eval不能替代no_grad，no_grad也不会自动关闭Dropout。

三个模型各训练25轮，每轮在同一验证mask上计算MAE。保存验证MAE最小的state_dict，平局较早轮；没有提前看83日评测段，也不会因中途结果多训练几轮。

种子42/43/44控制不同初始化；训练mask与shuffle各有独立固定种子，验证mask种子991对所有模型相同。固定版本、CPU线程和确定性算法让本次记录可复现，不承诺其他硬件/库版本逐位相同。

后续平均三模型的速度预测，不是只留下最好的种子，也不是把三个MAE平均当集成MAE。平均预测后再求绝对误差，才是集成的实际MAE。风险层还可使用三模型预测的标准差作为输入，但标准差不是已校准的不确定性区间。

## 论文中应怎样解释这一步

这一步提供缺测条件下的速度预测前端。验证数据参与选轮次，验证MAE降低不能当独立测试结论；速度预测更准也不保证事件告警更早或误报更少。后续必须冻结这三份权重，训练风险层、校准告警并一次评测，才能回答完整研究问题。

架构采用已有技术组合。论文可以解释实现、信息边界和实验结果，不能把这些基础PyTorch操作本身称为算法创新。
